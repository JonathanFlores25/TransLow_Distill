#!/usr/bin/env python3
"""
feature_extractor_GPU_1.py
==========================
Processes UCF-Crime videos into per-clip VideoMAE feature vectors (.npy).

Each output .npy contains:
    float32  (1, 768)  – mean-pooled VideoMAE last_hidden_state over patch tokens

The filename encodes the identity:
    {video_stem}_s{start:06d}.npy

Usage
-----
    python feature_extractor_GPU_1.py
    python feature_extractor_GPU_1.py --workers 2 --max-vram 40
    python feature_extractor_GPU_1.py --limit 10   # smoke-test
"""

from __future__ import annotations

import argparse
import gc
import logging
import logging.handlers
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parent
DATASET_TXT = _ROOT / "resources" / "Anomaly_Train_GPU_1.txt"
VIDEO_ROOT  = "/mnt/gpu02/datasets/Data_Arco_V2/UCF-Crime/videos"
OUTPUT_PATH = "/mnt/gpu02/datasets/Data_Arco_V2/UCF-Crime/features_videomae"

# ── VideoMAE ──────────────────────────────────────────────────────────────────
VIDEOMAE_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"

# ── Clip params ───────────────────────────────────────────────────────────────
CLIP_LEN  = 16   # frames per clip (VideoMAE expects exactly 16)
STRIDE    = 2    # temporal stride between consecutive clip frames
CLIP_STEP = 16   # step between clip start indices

# ── Multiprocessing defaults ──────────────────────────────────────────────────
NUM_WORKERS = 3
MAX_VRAM_GB = 50

# ── Category → output sub-directory (for organisation, not stored in file) ───
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


# ────────────────────────────────────────────────────────────────────────────
# Multiprocess-safe logging
# ────────────────────────────────────────────────────────────────────────────

class _WorkerPrefix(logging.Filter):
    def __init__(self, worker_id: int) -> None:
        super().__init__()
        self._pfx = f"W{worker_id}"

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = f"[{self._pfx}] {record.msg}"
        return True


def _start_listener(log_queue: "mp.Queue[Any]", log_dir: Path) -> logging.handlers.QueueListener:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh  = logging.FileHandler(log_dir / "feature_extractor.log", encoding="utf-8")
    sh  = logging.StreamHandler(sys.stderr)
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    return logging.handlers.QueueListener(log_queue, fh, sh, respect_handler_level=True)


def _attach_queue_logger(log_queue: "mp.Queue[Any]", worker_id: Optional[int] = None) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    if worker_id is not None:
        root.addFilter(_WorkerPrefix(worker_id))


# ────────────────────────────────────────────────────────────────────────────
# Frame utilities
# ────────────────────────────────────────────────────────────────────────────

def _read_frame(cap: cv2.VideoCapture, idx: int) -> Optional[np.ndarray]:
    """Seek to *idx* and return a BGR uint8 frame, or None on failure."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return frame if ret else None


def _bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC → RGB uint8 HWC (VideoMAEImageProcessor expects RGB)."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# ────────────────────────────────────────────────────────────────────────────
# Single video processing
# ────────────────────────────────────────────────────────────────────────────

def process_video(
    video_path: Path,
    category:   str,
    processor:  VideoMAEImageProcessor,
    model:      VideoMAEModel,
    device:     str,
    out_dir:    Path,
) -> int:
    """
    Extract VideoMAE features for every clip in one video and save as .npy.
    Returns number of NEW .npy files saved.

    Lazy-loading: only CLIP_LEN frames are decoded per clip (O(clip) RAM).
    """
    if not video_path.exists():
        logging.warning("VIDEO NOT FOUND – saltando: %s", video_path)
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.warning("No se puede abrir: %s", video_path)
        return 0

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info("[%s] %s  (%d frames)", category, video_path.name, total)

        min_frames = CLIP_LEN * STRIDE
        if total < min_frames:
            logging.warning(
                "[SKIP] Muy corto (%d/%d frames): %s",
                total, min_frames, video_path.name,
            )
            return 0

        out_dir.mkdir(parents=True, exist_ok=True)
        stem = video_path.stem

        clips_saved   = 0
        clips_skipped = 0

        for start in range(0, total - min_frames + 1, CLIP_STEP):
            out_file = out_dir / f"{stem}_s{start:06d}.npy"
            if out_file.exists():
                clips_skipped += 1
                continue

            src_idx = [start + i * STRIDE for i in range(CLIP_LEN)]

            # ── Lazy-load only the required frames ───────────────────────────
            raw_clip: List[np.ndarray] = []
            for idx in src_idx:
                frame = _read_frame(cap, idx)
                if frame is None:
                    break
                raw_clip.append(_bgr_to_rgb(frame))

            if len(raw_clip) < CLIP_LEN:
                logging.debug("Clip incompleto en start=%d, saltando", start)
                del raw_clip
                continue

            # ── VideoMAE inference ────────────────────────────────────────────
            with torch.no_grad():
                inputs = processor(raw_clip, return_tensors="pt").to(device)
                outputs = model(**inputs)
                # last_hidden_state: (1, num_patches, D) → mean → (1, D)
                features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            # ── Save as .npy — filename IS the identity ───────────────────────
            np.save(out_file, features.astype(np.float32))
            clips_saved += 1

            del raw_clip, inputs, outputs, features

        if clips_skipped and not clips_saved:
            logging.info("  -- %d clips ya procesados, nada nuevo", clips_skipped)
        else:
            skip_note = f"  (+{clips_skipped} ya existían)" if clips_skipped else ""
            logging.info("  OK  %d clips guardados%s", clips_saved, skip_note)

        return clips_saved

    finally:
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
    _attach_queue_logger(log_queue, worker_id)

    global CLIP_LEN, STRIDE, CLIP_STEP
    CLIP_LEN  = clip_len
    STRIDE    = stride
    CLIP_STEP = clip_step

    # ── VRAM budget per worker ────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory
        per_worker = (max_vram_gb / num_workers) * (1024 ** 3)
        frac = min(per_worker / total_vram, 0.95)
        torch.cuda.set_per_process_memory_fraction(frac)
        logging.info(
            "VRAM: fracción=%.2f  (%.1f GB / %.1f GB totales)",
            frac, frac * total_vram / (1024 ** 3), total_vram / (1024 ** 3),
        )

    # ── Load VideoMAE once per worker ─────────────────────────────────────────
    logging.info("Cargando VideoMAE (%s) en %s …", VIDEOMAE_MODEL, device)
    t0 = time.perf_counter()
    processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_MODEL)
    model     = VideoMAEModel.from_pretrained(VIDEOMAE_MODEL).to(device)
    model.eval()
    logging.info("Modelo listo en %.1f s", time.perf_counter() - t0)

    # ── Processing loop ───────────────────────────────────────────────────────
    total_clips   = 0
    vid_processed = 0
    vid_skipped   = 0
    t_start       = time.perf_counter()

    while True:
        item = video_queue.get()
        if item is None:  # poison pill
            break

        video_path, category, out_dir = item
        try:
            n = process_video(video_path, category, processor, model, device, out_dir)
            if n > 0:
                total_clips   += n
                vid_processed += 1
            else:
                vid_skipped += 1
        except Exception:
            logging.exception("Error procesando %s – el worker continúa", video_path.name)
            vid_skipped += 1
        finally:
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
        description="UCF-Crime → features VideoMAE (1, D) por clip en .npy"
    )
    p.add_argument("--txt",        type=Path,  default=DATASET_TXT)
    p.add_argument("--video-root", type=Path,  default=VIDEO_ROOT)
    p.add_argument("--output",     type=Path,  default=OUTPUT_PATH)
    p.add_argument("--model",      type=str,   default=VIDEOMAE_MODEL,
                   help="HuggingFace model id o path local")
    p.add_argument("--clip-len",   type=int,   default=CLIP_LEN)
    p.add_argument("--stride",     type=int,   default=STRIDE)
    p.add_argument("--clip-step",  type=int,   default=CLIP_STEP)
    p.add_argument("--workers",    type=int,   default=NUM_WORKERS)
    p.add_argument("--max-vram",   type=float, default=MAX_VRAM_GB)
    p.add_argument("--limit",      type=int,   default=0,
                   help="Procesar máximo N videos (0 = todos)")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    global VIDEOMAE_MODEL
    VIDEOMAE_MODEL = args.model

    with open(args.txt, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    items: List[Tuple[Path, str, Path]] = []
    for line in lines:
        parts = line.split("/", 1)
        if len(parts) < 2:
            continue
        category = CATEGORY_MAP.get(parts[0])
        if category is None:
            continue
        items.append((
            args.video_root / line,
            category,
            args.output / category,
        ))

    if args.limit:
        items = items[: args.limit]

    log_queue: mp.Queue = mp.Queue(-1)
    listener  = _start_listener(log_queue, args.output / "logs")
    listener.start()
    _attach_queue_logger(log_queue)

    t_main = time.perf_counter()
    logging.info("=" * 72)
    logging.info("VideoMAE Feature Extractor  |  UCF-Crime")
    logging.info("  Model       : %s", args.model)
    logging.info("  Workers     : %d", args.workers)
    logging.info("  VRAM límite : %.1f GB total  (%.1f GB/worker)",
                 args.max_vram, args.max_vram / args.workers)
    logging.info("  Videos      : %d", len(items))
    logging.info("  Clip params : len=%d  stride=%d  step=%d",
                 args.clip_len, args.stride, args.clip_step)
    logging.info("=" * 72)

    video_queue:   mp.Queue = mp.Queue()
    results_queue: mp.Queue = mp.Queue()

    for item in items:
        video_queue.put(item)
    for _ in range(args.workers):
        video_queue.put(None)

    workers = []
    for wid in range(args.workers):
        p = mp.Process(
            target=_worker_main,
            name=f"FEW{wid}",
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

    total_clips = total_processed = total_skipped = 0
    while not results_queue.empty():
        r = results_queue.get_nowait()
        total_clips     += r["clips"]
        total_processed += r["processed"]
        total_skipped   += r["skipped"]

    elapsed = time.perf_counter() - t_main
    logging.info("=" * 72)
    logging.info(
        "DONE en %.0f min  |  %d videos / %d saltados  |  %d clips totales",
        elapsed / 60, total_processed, total_skipped, total_clips,
    )
    logging.info("=" * 72)
    listener.stop()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
