#!/usr/bin/env python3
"""
FE_VideoMAE.py
==============
Extrae features VideoMAE de los videos UCF-Crime, produciendo un archivo
.npy por clip de 16 frames — con exactamente la misma segmentación temporal
que data_processor.py para que ambas fuentes se puedan unificar después.

Cada .npy contiene:
    embedding : float32  (768,)  – media sobre todos los patch tokens

Arquitectura
------------
  N reader threads (I/O) → queue → GPU batch processor (hilo principal)

  Los readers leen y decodifican videos en paralelo sin saturar RAM
  (buffer deslizante por video, máx ~48 frames vivos por thread).
  El procesador GPU drena la queue en batches para maximizar utilización.

Uso
---
    python FE_VideoMAE.py
    python FE_VideoMAE.py --workers 4 --batch-size 32
    python FE_VideoMAE.py --limit 10    # smoke-test
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_TXT = _ROOT / "resources" / "Anomaly_Train.txt"
VIDEO_ROOT  = Path("/media/pc/backup1/BaseDeDatos/UCF-Crime/Videos")
OUTPUT_PATH = _ROOT / "data" / "vmae_features"

# ── Clip params — deben coincidir con data_processor.py ──────────────────────
CLIP_LEN    = 16   # frames por clip
STRIDE      = 2    # stride temporal entre frames del clip
CLIP_STEP   = 16   # paso entre inicios de clips consecutivos
OUTPUT_SIZE = 224

# ── Defaults de rendimiento ───────────────────────────────────────────────────
NUM_READERS = 4    # threads de lectura/decodificación (I/O bound)
BATCH_SIZE  = 32   # clips por forward pass de VideoMAE (~4-6 GB VRAM)
QUEUE_MAX   = 512  # máx clips pendientes en la queue

# ── Modelo ────────────────────────────────────────────────────────────────────
MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"

# ── Mapeo de categorías (igual que data_processor.py) ────────────────────────
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

_SENTINEL = object()  # poison pill para la queue


# ────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────

def setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logging.getLogger().addHandler(sh)
    logging.getLogger().setLevel(logging.INFO)


# ────────────────────────────────────────────────────────────────────────────
# Utilidades de frames
# ────────────────────────────────────────────────────────────────────────────

def resize224(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(frame_bgr, (OUTPUT_SIZE, OUTPUT_SIZE))


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# ────────────────────────────────────────────────────────────────────────────
# Reader thread  (I/O + decode, sin GPU)
# ────────────────────────────────────────────────────────────────────────────

def _reader_worker(
    worker_id: int,
    items:     List[Tuple[Path, str, Path]],
    queue:     "Queue[object]",
    clip_len:  int,
    stride:    int,
    clip_step: int,
) -> None:
    """
    Lee videos, extrae clips con buffer deslizante y los encola.
    Cada item de la queue es (out_path, frames_rgb) o _SENTINEL al final.
    RAM usada: ≤ clip_len*stride + clip_step frames por video (~48 frames).
    """
    for video_path, expert_type, out_dir in items:
        if not video_path.exists():
            logging.warning("[R%d] NOT FOUND: %s", worker_id, video_path)
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.warning("[R%d] No se puede abrir: %s", worker_id, video_path)
            continue

        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        min_frames = clip_len * stride

        if 0 < total < min_frames:
            cap.release()
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        stem: str                       = video_path.stem
        frame_buffer: Dict[int, np.ndarray] = {}
        next_clip_start = 0
        frame_idx       = 0
        real_total      = 0

        while True:
            ret, raw = cap.read()
            if not ret:
                break

            frame_buffer[frame_idx] = bgr_to_rgb(resize224(raw))
            frame_idx  += 1
            real_total += 1

            # Encolar todos los clips que ya tienen sus frames completos
            while True:
                clip_end = next_clip_start + (clip_len - 1) * stride
                if clip_end >= frame_idx:
                    break

                out_path = out_dir / f"{stem}_s{next_clip_start:06d}.npy"
                if not out_path.exists():
                    src_idx    = [next_clip_start + i * stride for i in range(clip_len)]
                    frames_rgb = [frame_buffer[i] for i in src_idx]
                    queue.put((out_path, frames_rgb))  # bloquea si queue está llena

                next_clip_start += clip_step

                # Evictar frames ya no necesarios
                for k in [k for k in frame_buffer if k < next_clip_start]:
                    del frame_buffer[k]

        cap.release()

        if real_total < min_frames:
            logging.warning("[R%d] Muy corto (%d frames): %s",
                            worker_id, real_total, video_path.name)
            continue

        logging.info("[R%d] %s  → %d frames leídos", worker_id, video_path.name, real_total)

    queue.put(_SENTINEL)


# ────────────────────────────────────────────────────────────────────────────
# GPU batch processor  (hilo principal)
# ────────────────────────────────────────────────────────────────────────────

def _gpu_processor(
    queue:      "Queue[object]",
    processor:  VideoMAEImageProcessor,
    model:      VideoMAEModel,
    device:     str,
    batch_size: int,
    n_readers:  int,
) -> int:
    """
    Drena la queue en batches, corre VideoMAE y guarda .npy.
    Retorna el total de clips guardados.
    """
    sentinels   = 0
    total_saved = 0

    batch_paths:  List[Path]                  = []
    batch_frames: List[List[np.ndarray]]      = []

    def flush() -> None:
        nonlocal total_saved
        if not batch_paths:
            return
        inputs = processor(batch_frames, return_tensors="pt").to(device)
        if device == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}
        with torch.no_grad():
            outputs    = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # [B, 768]
        emb_np = embeddings.cpu().float().numpy().astype(np.float32)
        for path, emb in zip(batch_paths, emb_np):
            np.save(path, emb)
            total_saved += 1
        logging.info("  [GPU] batch %d clips guardados  (total=%d)",
                     len(batch_paths), total_saved)
        batch_paths.clear()
        batch_frames.clear()

    while True:
        try:
            item = queue.get(timeout=2.0)
        except Empty:
            if sentinels >= n_readers:
                flush()
                break
            flush()  # vaciar batch parcial mientras esperamos más clips
            continue

        if item is _SENTINEL:
            sentinels += 1
            if sentinels >= n_readers:
                flush()
                break
            continue

        out_path, frames_rgb = item  # type: ignore[misc]
        batch_paths.append(out_path)
        batch_frames.append(frames_rgb)

        if len(batch_frames) >= batch_size:
            flush()

    return total_saved


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UCF-Crime → VideoMAE embeddings .npy  (16 frames/clip)"
    )
    p.add_argument("--txt",        type=Path, default=DATASET_TXT)
    p.add_argument("--video-root", type=Path, default=VIDEO_ROOT)
    p.add_argument("--output",     type=Path, default=OUTPUT_PATH)
    p.add_argument("--model",      type=str,  default=MODEL_NAME)
    p.add_argument("--clip-len",   type=int,  default=CLIP_LEN)
    p.add_argument("--stride",     type=int,  default=STRIDE)
    p.add_argument("--clip-step",  type=int,  default=CLIP_STEP)
    p.add_argument("--workers",    type=int,  default=NUM_READERS,
                   help="Threads de lectura de video (default 4)")
    p.add_argument("--batch-size", type=int,  default=BATCH_SIZE,
                   help="Clips por forward pass GPU (default 32, ~5 GB VRAM). "
                        "Subir a 48-64 si tienes 12+ GB libres.")
    p.add_argument("--queue-max",  type=int,  default=QUEUE_MAX,
                   help="Máx clips en cola antes de bloquear readers (default 512)")
    p.add_argument("--limit",      type=int,  default=0,
                   help="Procesar máximo N videos (0 = todos)")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    args = parse_args()

    with open(args.txt, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    items: List[Tuple[Path, str, Path]] = []
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
            args.output / expert_type,
        ))

    if args.limit:
        items = items[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("=" * 72)
    logging.info("VideoMAE Feature Extractor  |  UCF-Crime")
    logging.info("  Modelo       : %s", args.model)
    logging.info("  Device       : %s", device)
    logging.info("  Videos       : %d", len(items))
    logging.info("  Clip params  : len=%d  stride=%d  step=%d",
                 args.clip_len, args.stride, args.clip_step)
    logging.info("  Reader threads: %d", args.workers)
    logging.info("  Batch size   : %d clips/forward", args.batch_size)
    logging.info("  Salida       : %s", args.output)
    logging.info("=" * 72)

    # ── Cargar modelo ─────────────────────────────────────────────────────────
    logging.info("Cargando VideoMAE …")
    t0        = time.perf_counter()
    processor = VideoMAEImageProcessor.from_pretrained(args.model)
    model     = VideoMAEModel.from_pretrained(args.model).to(device)
    if device == "cuda":
        model = model.half()                  # fp16: ~2x throughput, mitad de VRAM
        model = torch.compile(model)          # JIT fusion: +20-30% adicional
    model.eval()
    logging.info("Modelo listo en %.1f s", time.perf_counter() - t0)

    # ── Repartir videos entre readers ─────────────────────────────────────────
    n = args.workers
    chunks: List[List[Tuple[Path, str, Path]]] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        chunks[i % n].append(item)

    # ── Lanzar reader threads ─────────────────────────────────────────────────
    queue: Queue = Queue(maxsize=args.queue_max)
    threads = []
    for wid, chunk in enumerate(chunks):
        t = threading.Thread(
            target=_reader_worker,
            args=(wid, chunk, queue, args.clip_len, args.stride, args.clip_step),
            name=f"Reader-{wid}",
            daemon=True,
        )
        t.start()
        threads.append(t)

    # ── GPU processor en hilo principal ──────────────────────────────────────
    t_start     = time.perf_counter()
    total_saved = _gpu_processor(
        queue, processor, model, device, args.batch_size, n
    )

    for t in threads:
        t.join()

    elapsed = time.perf_counter() - t_start
    logging.info("=" * 72)
    logging.info(
        "DONE en %.0f min  |  %d clips guardados",
        elapsed / 60, total_saved,
    )
    logging.info("=" * 72)


if __name__ == "__main__":
    main()
