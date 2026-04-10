#!/usr/bin/env python3
"""
data_processor.py
=================
Processes UCF-Crime videos into 16-frame .npz clips annotated with teacher
attention masks and anomaly scores for Video Anomaly Detection distillation.

Each output .npz contains:
    video     : float32  (16, 3, 224, 224)  – RGB, normalised to [0, 1]
    attn_mask : float32  (224, 224)         – spatial attention from teacher expert
    label     : float32  scalar             – 1.0 anomaly / 0.0 normal

Expert routing
--------------
    Fighting / Assault   → FightExpert
    RoadAccidents        → CrashExpert
    Arson / Explosion    → FireExpert
    Burglary / Robbery   → RobberyExpert
    Stealing             → CarPartsExpert
    Normal               → all experts (dispersed negative examples)

Usage
-----
    python data_processor.py                          # defaults: 3 workers, 11.5 GB VRAM
    python data_processor.py --workers 4 --max-vram 13
    python data_processor.py --limit 10              # smoke-test on 10 videos
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import logging
import logging.handlers
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2  # type: ignore[import-untyped]
import clip  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]
import torch  # type: ignore[import-untyped]
from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]

# ── Project root ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from core.base_expert import BaseExpert
from core.tracker import ArconteTracker
from experts_V2_0_0.carparts_expert import CarPartsExpert, _compute_pc as _carparts_compute_pc
from experts_V2_0_0.crash_expert import CrashExpert
from experts_V2_0_0.fight_expert import FightExpert
from experts_V2_0_0.fire_expert import FireExpert, make_fire_smoke_experts
from experts_V2_0_0.robbery_expert import RobberyExpert

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_TXT   = _ROOT / "resources" / "Anomaly_Train_GPU_1.txt"
VIDEO_ROOT    = "/mnt/gpu02/datasets/Data_Arco_V2/UCF-Crime/videos"
OUTPUT_PATH   = "/mnt/gpu02/datasets/Data_Arco_V2/UCF-Crime/processed"
CLIP_CKPT     = _ROOT / ".checkpoints" / "ViT-L-14.pt"
ROBBERY_CKPT  = _ROOT / ".checkpoints" / "ViT-B-16-32-f.pt"

# ── Clip params ───────────────────────────────────────────────────────────────
CLIP_LEN    = 16    # frames per clip
STRIDE      = 2     # temporal stride between consecutive clip frames
CLIP_STEP   = 16    # step between clip start indices
OUTPUT_SIZE = 224
BLUR_SIGMA  = 15.0

# ── Multiprocessing defaults ──────────────────────────────────────────────────
NUM_WORKERS = 3       # parallel worker processes
MAX_VRAM_GB = 50    # total VRAM budget across all workers (leave 4.5 GB headroom)

# ── Category mapping ──────────────────────────────────────────────────────────
CATEGORY_MAP: Dict[str, str] = {
    "Fighting":      "fight",
    "Assault":       "fight",
    "RoadAccidents": "crash",
    "Arson":         "fire",
    "Explosion":     "fire",
    "Burglary":      "robbery",
    "Robbery":       "robbery",
    "Stealing":      "carparts",
    "Normal":        "normal",
}

LABEL_MAP: Dict[str, float] = {
    "fight":    1.0,
    "crash":    1.0,
    "fire":     1.0,
    "robbery":  1.0,
    "carparts": 1.0,
    "normal":   0.0,
}


# ────────────────────────────────────────────────────────────────────────────
# Stdout suppressor  (silences print() calls inside expert classes)
# ────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _quiet() -> Generator[None, None, None]:
    """Redirect stdout → /dev/null for the duration of the block."""
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


# ────────────────────────────────────────────────────────────────────────────
# Multiprocess-safe logging
# ────────────────────────────────────────────────────────────────────────────

class _WorkerPrefix(logging.Filter):
    """Prepends [Wn] to every log record emitted by a worker process."""

    def __init__(self, worker_id: int) -> None:
        super().__init__()
        self._pfx = f"W{worker_id}"

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = f"[{self._pfx}] {record.msg}"
        return True


def _start_listener(
    log_queue: "mp.Queue[Any]",
    log_dir: Path,
) -> logging.handlers.QueueListener:
    """
    Create (but do not start) a QueueListener in the main process.
    All worker records funnel through the queue, so file writes are serialised.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_dir / "data_processor.log", encoding="utf-8")
    sh = logging.StreamHandler(sys.stderr)
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    return logging.handlers.QueueListener(log_queue, fh, sh, respect_handler_level=True)


def _attach_queue_logger(
    log_queue: "mp.Queue[Any]",
    worker_id: Optional[int] = None,
) -> None:
    """
    Route the root logger for the calling process through the shared queue.
    If worker_id is given, a prefix filter is installed so log lines are tagged.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    if worker_id is not None:
        root.addFilter(_WorkerPrefix(worker_id))


# ────────────────────────────────────────────────────────────────────────────
# Frame utilities
# ────────────────────────────────────────────────────────────────────────────

def bgr_to_chw_float(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC → RGB float32 CHW in [0, 1]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1).astype(np.float32) / 255.0


def resize224(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (OUTPUT_SIZE, OUTPUT_SIZE))


# ────────────────────────────────────────────────────────────────────────────
# Attention mask helpers
# ────────────────────────────────────────────────────────────────────────────

def _xyxy_bboxes_to_mask(
    bboxes_xyxy: np.ndarray,
    frame_wh: Tuple[int, int],
    out: int = OUTPUT_SIZE,
    sigma: float = BLUR_SIGMA,
) -> np.ndarray:
    W, H = frame_wh
    mask = np.zeros((out, out), dtype=np.float32)
    if len(bboxes_xyxy) == 0:
        return mask
    sx, sy = out / max(W, 1), out / max(H, 1)
    for bb in bboxes_xyxy:
        x1, y1, x2, y2 = bb[:4]
        lx = int(np.clip(x1 * sx, 0, out - 1))
        ly = int(np.clip(y1 * sy, 0, out - 1))
        rx = int(np.clip(x2 * sx, 1, out))
        ry = int(np.clip(y2 * sy, 1, out))
        mask[ly:ry, lx:rx] = 1.0
    if mask.max() > 0:
        mask = gaussian_filter(mask, sigma=sigma)
        mask /= mask.max()
    return mask


def _xywh_bboxes_to_mask(
    bboxes_xywh: List[Tuple[Any, ...]],
    frame_wh: Tuple[int, int],
    out: int = OUTPUT_SIZE,
    sigma: float = BLUR_SIGMA,
) -> np.ndarray:
    if not bboxes_xywh:
        return np.zeros((out, out), dtype=np.float32)
    arr = np.array(
        [[x, y, x + w, y + h] for x, y, w, h in bboxes_xywh],
        dtype=np.float32,
    )
    return _xyxy_bboxes_to_mask(arr, frame_wh, out, sigma)


def extract_attention(
    expert: BaseExpert,
    last_tracker_data: Dict[str, Any],
    frame_wh: Tuple[int, int],
) -> np.ndarray:
    """
    Return a (OUTPUT_SIZE × OUTPUT_SIZE) float32 attention mask.

    CarPartsExpert  → M_norm from _compute_pc on the stored key frame.
    FireExpert      → Gaussian mask from fire detector bboxes (x, y, w, h).
    Fight/Robbery   → Gaussian mask from tracker person bboxes.
    CrashExpert     → Gaussian mask from tracker vehicle bboxes.
    """
    out = OUTPUT_SIZE

    if isinstance(expert, CarPartsExpert):
        kf = expert._last_kf_bgr
        if kf is not None and kf.size > 0:
            try:
                with _quiet():
                    M_norm, _, _, _ = _carparts_compute_pc(
                        kf, expert._text_feat, expert._model, expert._preprocess,
                    )
                if M_norm is not None and M_norm.size > 0:
                    m = cv2.resize(
                        M_norm.astype(np.float32), (out, out),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    if m.max() > 0:
                        return m / m.max()
            except Exception as exc:
                logging.debug("CarParts _compute_pc failed: %s", exc)
        return np.zeros((out, out), dtype=np.float32)

    if isinstance(expert, FireExpert):
        try:
            bboxes = expert.get_display_data().get("bboxes", [])
            if bboxes:
                return _xywh_bboxes_to_mask(bboxes, frame_wh, out)
        except Exception:
            pass
        return np.ones((out, out), dtype=np.float32)

    if isinstance(expert, (FightExpert, RobberyExpert)):
        persons = last_tracker_data.get("persons_xyxy", np.empty((0, 4), dtype=np.float32))
        mask = _xyxy_bboxes_to_mask(persons, frame_wh, out)
        return mask if mask.max() > 0 else np.full((out, out), 0.1, dtype=np.float32)

    if isinstance(expert, CrashExpert):
        vehicles = last_tracker_data.get("vehicles_xyxy", np.empty((0, 4), dtype=np.float32))
        mask = _xyxy_bboxes_to_mask(vehicles, frame_wh, out)
        return mask if mask.max() > 0 else np.full((out, out), 0.1, dtype=np.float32)

    return np.full((out, out), 0.1, dtype=np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Normal video helpers
# ────────────────────────────────────────────────────────────────────────────

def process_normal_clip(
    experts: Dict[str, BaseExpert],
    frames: List[np.ndarray],
    last_tracker_data: Dict[str, Any],
    frame_wh: Tuple[int, int],
) -> Tuple[float, np.ndarray]:
    """Run all experts on a Normal clip → (mean_score, averaged_mask)."""
    scores: List[float] = []
    masks:  List[np.ndarray] = []
    for expert in experts.values():
        try:
            with _quiet():
                result = expert.predict(frames)
            scores.append(float(result.get("score", 0.0)))
        except Exception:
            scores.append(0.0)
        masks.append(extract_attention(expert, last_tracker_data, frame_wh))
    mean_score = float(np.mean(scores)) if scores else 0.0
    combined   = np.mean(np.stack(masks, axis=0), axis=0).astype(np.float32)
    if combined.max() > 0:
        combined /= combined.max()
    return mean_score, combined


# ────────────────────────────────────────────────────────────────────────────
# Single video processing
# ────────────────────────────────────────────────────────────────────────────

def _read_frame(cap: cv2.VideoCapture, idx: int) -> Optional[np.ndarray]:
    """Seek to *idx* and return the decoded BGR frame, or None on failure."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return frame if ret else None


def process_video(
    video_path:  Path,
    expert_type: str,
    experts:     Dict[str, BaseExpert],
    tracker:     ArconteTracker,
    out_dir:     Path,
    label:       float,
) -> int:
    """
    Extract and save .npz clips from one video.
    Returns the number of NEW clips saved (0 = skipped or error).

    Lazy-loading: only the 16 frames needed per clip are decoded at a time,
    so RAM usage is O(clip_size) instead of O(video_size).
    """
    if not video_path.exists():
        logging.warning("VIDEO NOT FOUND – saltando: %s", video_path)
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.warning("No se puede abrir: %s", video_path)
        return 0

    try:
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_wh = (fw, fh)

        logging.info("[%s] %s  (%d frames)", expert_type, video_path.name, total)

        min_frames = CLIP_LEN * STRIDE
        if total < min_frames:
            logging.warning(
                "[SKIP] Muy corto (%d/%d frames): %s",
                total, min_frames, video_path.name,
            )
            return 0

        out_dir.mkdir(parents=True, exist_ok=True)
        stem   = video_path.stem
        active = experts if expert_type == "normal" else {expert_type: experts[expert_type]}

        clips_saved   = 0
        clips_skipped = 0
        scores_accum: List[float] = []

        for start in range(0, total - min_frames + 1, CLIP_STEP):
            clip_name = f"{stem}_s{start:06d}.npz"
            if (out_dir / clip_name).exists():
                clips_skipped += 1
                continue

            src_idx = [start + i * STRIDE for i in range(CLIP_LEN)]

            # ── Lazy-load only the 16 frames required for this clip ──────────
            raw_clip: List[np.ndarray] = []
            for idx in src_idx:
                frame = _read_frame(cap, idx)
                if frame is None:
                    break
                raw_clip.append(frame)

            if len(raw_clip) < CLIP_LEN:
                logging.debug("Clip incompleto en start=%d, saltando", start)
                del raw_clip
                continue

            # ── All inference inside no_grad to avoid graph accumulation ─────
            with torch.no_grad():
                # Feed frames through tracker + heuristics to build expert state
                last_td: Dict[str, Any] = {}
                for frame in raw_clip:
                    try:
                        with _quiet():
                            td = tracker.process(frame)
                        last_td = td
                    except Exception:
                        td = last_td

                    for expert in active.values():
                        try:
                            with _quiet():
                                expert.process_heuristics(frame, td)
                        except Exception:
                            pass

                resized = [resize224(f) for f in raw_clip]

                if expert_type == "normal":
                    score, attn_mask = process_normal_clip(active, resized, last_td, frame_wh)
                else:
                    expert = active[expert_type]
                    try:
                        with _quiet():
                            result = expert.predict(resized)
                        score = float(result.get("score", 0.0))
                    except Exception:
                        score = 0.0
                    attn_mask = extract_attention(expert, last_td, frame_wh)

            video_arr = np.stack(
                [bgr_to_chw_float(f) for f in resized], axis=0,
            ).astype(np.float32)

            np.savez_compressed(
                out_dir / clip_name,
                video=video_arr,
                attn_mask=attn_mask.astype(np.float32),
                label=np.float32(label),
            )
            clips_saved  += 1
            scores_accum.append(score)

            # ── Release per-clip buffers immediately ─────────────────────────
            del raw_clip, resized, video_arr, attn_mask

        mean_s = float(np.mean(scores_accum)) if scores_accum else 0.0
        max_s  = float(np.max(scores_accum))  if scores_accum else 0.0

        if clips_skipped and not clips_saved:
            logging.info("  -- %d clips ya procesados, nada nuevo", clips_skipped)
        else:
            skip_note = f"  (+{clips_skipped} ya existían)" if clips_skipped else ""
            logging.info(
                "  OK  %d clips guardados%s | score avg=%.3f max=%.3f | label=%.0f",
                clips_saved, skip_note, mean_s, max_s, label,
            )
        return clips_saved

    finally:
        # Always release the VideoCapture handle, even on exception
        cap.release()


# ────────────────────────────────────────────────────────────────────────────
# Worker process entry point
# ────────────────────────────────────────────────────────────────────────────

def _worker_main(
    worker_id:     int,
    video_queue:   "mp.Queue[Any]",
    log_queue:     "mp.Queue[Any]",
    results_queue: "mp.Queue[Any]",
    clip_len:      int,
    stride:        int,
    clip_step:     int,
    num_workers:   int,
    max_vram_gb:   float,
) -> None:
    # ── Logging via shared queue ──────────────────────────────────────────────
    _attach_queue_logger(log_queue, worker_id)

    # ── Propagate clip params (each spawned process has its own globals) ──────
    global CLIP_LEN, STRIDE, CLIP_STEP
    CLIP_LEN  = clip_len
    STRIDE    = stride
    CLIP_STEP = clip_step

    # ── VRAM limit — must be called before any CUDA allocation ────────────────
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory  # bytes
        per_worker = (max_vram_gb / num_workers) * (1024 ** 3)
        frac = min(per_worker / total_vram, 0.95)
        torch.cuda.set_per_process_memory_fraction(frac)
        logging.info(
            "VRAM: fracción=%.2f  (%.1f GB asignados de %.1f GB totales)",
            frac,
            frac * total_vram / (1024 ** 3),
            total_vram / (1024 ** 3),
        )

    # ── Load models ───────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Cargando CLIP ViT-L/14 en %s …", device)
    t0 = time.perf_counter()

    with _quiet():
        clip_model, clip_preprocess = clip.load(str(CLIP_CKPT), device=device, jit=False)
    clip_model.eval()

    fire_expert, _ = make_fire_smoke_experts()
    experts: Dict[str, BaseExpert] = {
        "fight":    FightExpert(),
        "crash":    CrashExpert(),
        "fire":     fire_expert,
        "robbery":  RobberyExpert(checkpoint_path=str(ROBBERY_CKPT)),
        "carparts": CarPartsExpert(),
    }
    with _quiet():
        for expert in experts.values():
            expert.load(clip_model, clip_preprocess)

    with _quiet():
        tracker = ArconteTracker()

    logging.info("Modelos listos en %.1f s", time.perf_counter() - t0)

    # ── Processing loop ───────────────────────────────────────────────────────
    total_clips   = 0
    vid_processed = 0
    vid_skipped   = 0
    t_start       = time.perf_counter()

    while True:
        item = video_queue.get()
        if item is None:   # poison pill → worker done
            break

        video_path, expert_type, label, out_dir = item
        try:
            n = process_video(video_path, expert_type, experts, tracker, out_dir, label)
            if n > 0:
                total_clips   += n
                vid_processed += 1
            else:
                vid_skipped += 1
        except Exception:
            logging.exception("Error procesando %s – el worker continúa", video_path.name)
            vid_skipped += 1
        finally:
            # Aggressive cleanup after every video to prevent memory leaks
            # that accumulate over hours/days of continuous processing.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.perf_counter() - t_start
    logging.info(
        "Worker terminado en %.0f min | %d videos OK / %d saltados | %d clips",
        elapsed / 60, vid_processed, vid_skipped, total_clips,
    )
    results_queue.put({
        "worker_id": worker_id,
        "clips":     total_clips,
        "processed": vid_processed,
        "skipped":   vid_skipped,
    })


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UCF-Crime → clips .npz con etiquetas de maestro Arconte"
    )
    p.add_argument("--txt",        type=Path,  default=DATASET_TXT)
    p.add_argument("--video-root", type=Path,  default=VIDEO_ROOT)
    p.add_argument("--output",     type=Path,  default=OUTPUT_PATH)
    p.add_argument("--clip-len",   type=int,   default=CLIP_LEN)
    p.add_argument("--stride",     type=int,   default=STRIDE)
    p.add_argument("--clip-step",  type=int,   default=CLIP_STEP)
    p.add_argument("--workers",    type=int,   default=NUM_WORKERS,
                   help="Procesos paralelos (default 3)")
    p.add_argument("--max-vram",   type=float, default=MAX_VRAM_GB,
                   help="VRAM total en GB repartida entre workers (default 11.5)")
    p.add_argument("--limit",      type=int,   default=0,
                   help="Procesar máximo N videos (0 = todos)")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Build ordered video list from TXT ────────────────────────────────────
    with open(args.txt, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    items: List[Tuple[Path, str, float, Path]] = []
    for line in lines:
        parts = line.split("/", 1)
        if len(parts) < 2:
            continue
        category    = parts[0]
        expert_type = CATEGORY_MAP.get(category)
        if expert_type is None:
            continue
        items.append((
            args.video_root / line,
            expert_type,
            LABEL_MAP[expert_type],
            args.output / expert_type,
        ))

    if args.limit:
        items = items[: args.limit]

    # ── Logging: all processes funnel through one queue → serialised writes ───
    log_queue: mp.Queue = mp.Queue(-1)          # type: ignore[type-arg]
    listener  = _start_listener(log_queue, args.output / "logs")
    listener.start()
    _attach_queue_logger(log_queue)             # main process uses same queue

    t_main = time.perf_counter()
    logging.info("=" * 72)
    logging.info("Arconte-Distill  |  UCF-Crime data processor")
    logging.info("  Workers     : %d", args.workers)
    logging.info("  VRAM límite : %.1f GB total  (%.1f GB/worker)",
                 args.max_vram, args.max_vram / args.workers)
    logging.info("  Videos      : %d", len(items))
    logging.info("  Clip params : len=%d  stride=%d  step=%d",
                 args.clip_len, args.stride, args.clip_step)
    logging.info("=" * 72)

    # ── Fill video queue ──────────────────────────────────────────────────────
    video_queue:   mp.Queue = mp.Queue()        # type: ignore[type-arg]
    results_queue: mp.Queue = mp.Queue()        # type: ignore[type-arg]

    for item in items:
        video_queue.put(item)
    for _ in range(args.workers):
        video_queue.put(None)                   # one poison pill per worker

    # ── Spawn workers ─────────────────────────────────────────────────────────
    workers = []
    for wid in range(args.workers):
        p = mp.Process(
            target=_worker_main,
            name=f"ArcW{wid}",
            args=(
                wid,
                video_queue,
                log_queue,
                results_queue,
                args.clip_len,
                args.stride,
                args.clip_step,
                args.workers,
                args.max_vram,
            ),
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    # ── Aggregate totals ──────────────────────────────────────────────────────
    total_clips = total_processed = total_skipped = 0
    while not results_queue.empty():
        r = results_queue.get_nowait()
        total_clips     += r["clips"]
        total_processed += r["processed"]
        total_skipped   += r["skipped"]

    elapsed = time.perf_counter() - t_main
    logging.info("=" * 72)
    logging.info(
        "DONE en %.0f min  |  %d videos procesados / %d saltados  |  %d clips totales",
        elapsed / 60, total_processed, total_skipped, total_clips,
    )
    logging.info("=" * 72)

    listener.stop()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()