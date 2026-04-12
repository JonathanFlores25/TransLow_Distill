#!/usr/bin/env python3
"""
FE_VideoMAE.py
==============
Extrae features VideoMAE de los videos UCF-Crime, produciendo un archivo
.npy por clip de 16 frames.

Modos de extracción
--------------------
  Modo estándar (default):
      shape   : float32  (768,)
      tamaño  : ~3 KB/clip
      uso     : videoMae_Freeze v1/v2 — clasificador FC + Soft Triplet

  Modo espacial (--spatial):
      shape   : float16  (8, 14, 14, 768)
                T=8   = CLIP_LEN(16) / tubelet_size(2)
                H=W=14 = OUTPUT_SIZE(224) / patch_size(16)
                D=768  = hidden_size ViT-Base
      tamaño  : ~2.4 MB/clip  (~1.75 TB para los 732K clips de UCF-Crime)
      uso     : videoMae_Freeze v3 — Factored Attention (espacial + temporal)

      Nota de VRAM: batch-size 32 ocupa ~18 GB con spatial.
      Recomendado: --batch-size 8 (RTX 3080/3090) o --batch-size 16 (A100).

Arquitectura
------------
  N reader threads (I/O) → queue → GPU batch processor (hilo principal)

Uso
---
    python FE_VideoMAE.py                                     # estándar
    python FE_VideoMAE.py --spatial                           # espacial
    python FE_VideoMAE.py --spatial --txt /media/pc/MainWork/Codes/TransLow_Distill/resources/Anomaly_Train_GPU_1.txt --output /media/pc/backup1/BaseDeDatos/UCF-Crime/Features_S_VideoMAE   # disco externo
    python FE_VideoMAE.py --spatial --batch-size 8            # batch menor (VRAM)
    python FE_VideoMAE.py --limit 10                          # smoke-test
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
DATASET_TXT          = _ROOT / "resources" / "Anomaly_Train.txt"
VIDEO_ROOT           = Path("/media/pc/backup1/BaseDeDatos/UCF-Crime/Videos")
OUTPUT_PATH          = _ROOT / "data" / "vmae_features"
OUTPUT_PATH_SPATIAL  = _ROOT / "data" / "vmae_features_spatial"

# ── Clip params — deben coincidir con data_processor.py ──────────────────────
CLIP_LEN    = 16   # frames por clip
STRIDE      = 2    # stride temporal entre frames del clip
CLIP_STEP   = 16   # paso entre inicios de clips consecutivos
OUTPUT_SIZE = 224

# ── Defaults de rendimiento ───────────────────────────────────────────────────
# 4090 24 GB: batch_size=96 ocupa ~18 GB en modo estándar (fp16).
# Referencia orientativa para ajustar:
#   batch_size 32  →  ~5-6 GB    RTX 3070/3080 8-10 GB
#   batch_size 64  →  ~9-11 GB   RTX 3080 10 GB / 3090
#   batch_size 96  →  ~16-18 GB  RTX 4090 24 GB  ← default
#   batch_size 128 →  ~22-24 GB  A100 80 GB
NUM_READERS = 8    # threads de lectura/decodificación (I/O bound) — 8 en NVMe/SSD
BATCH_SIZE  = 96   # clips por forward pass de VideoMAE
QUEUE_MAX   = 1024 # máx clips pendientes en la queue (mantiene GPU ocupada)

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
    spatial:    bool = False,
) -> int:
    """
    Drena la queue en batches, corre VideoMAE y guarda .npy.

    Modos
    -----
    spatial=False  → mean-pool last_hidden_state → (768,) float32
    spatial=True   → reshape last_hidden_state   → (8, 14, 14, 768) float16
                     T=8 (CLIP_LEN/tubelet_size=16/2)
                     H=W=14 (OUTPUT_SIZE/patch_size=224/16)

    Retorna el total de clips guardados.
    """
    # Tokens esperados: T * H * W = 8 * 14 * 14 = 1568
    _T, _H, _W = 8, 14, 14
    assert _T * _H * _W == 1568, "Revisar tubelet/patch sizes"

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
            outputs = model(**inputs)
            hidden  = outputs.last_hidden_state  # [B, 1568, 768]

        if spatial:
            # Reshape a (B, T, H, W, D) y guardar como float16
            B = hidden.shape[0]
            tokens = hidden.reshape(B, _T, _H, _W, 768)  # [B, 8, 14, 14, 768]
            feat_np = tokens.cpu().to(torch.float16).numpy()
            for path, feat in zip(batch_paths, feat_np):
                np.save(path, feat)          # float16, (8, 14, 14, 768)
                total_saved += 1
        else:
            # Mean-pool y guardar como float32
            embeddings = hidden.mean(dim=1)  # [B, 768]
            emb_np = embeddings.cpu().float().numpy().astype(np.float32)
            for path, emb in zip(batch_paths, emb_np):
                np.save(path, emb)           # float32, (768,)
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
    p.add_argument("--output",     type=Path, default=None,
                   help="Directorio de salida. "
                        "Default: data/vmae_features (estándar) o "
                        "data/vmae_features_spatial (--spatial).")
    p.add_argument("--model",      type=str,  default=MODEL_NAME)
    p.add_argument("--clip-len",   type=int,  default=CLIP_LEN)
    p.add_argument("--stride",     type=int,  default=STRIDE)
    p.add_argument("--clip-step",  type=int,  default=CLIP_STEP)
    p.add_argument("--workers",    type=int,  default=NUM_READERS,
                   help="Threads de lectura de video (default 4)")
    p.add_argument("--batch-size", type=int,  default=BATCH_SIZE,
                   help="Clips por forward pass GPU "
                        "(default 32, ~5 GB VRAM estándar; "
                        "usar 8-16 con --spatial para evitar OOM).")
    p.add_argument("--queue-max",  type=int,  default=QUEUE_MAX,
                   help="Máx clips en cola antes de bloquear readers (default 512)")
    p.add_argument("--limit",      type=int,  default=0,
                   help="Procesar máximo N videos (0 = todos)")
    p.add_argument("--spatial",    action="store_true",
                   help="Guardar tokens espaciales (8,14,14,768) float16 "
                        "en lugar del embedding global (768,) float32. "
                        "Requiere ~2.4 MB/clip. Recomendado --batch-size 16.")
    # ── Sharding: para correr múltiples instancias en paralelo ───────────────
    # Terminal 0:  --num-shards 2 --shard-id 0
    # Terminal 1:  --num-shards 2 --shard-id 1
    # Cada instancia procesa su mitad de la lista de videos sin solaparse.
    p.add_argument("--num-shards", type=int, default=1,
                   help="Número total de shards (instancias paralelas). "
                        "Default 1 = sin sharding.")
    p.add_argument("--shard-id",   type=int, default=0,
                   help="Índice del shard actual (0-based). "
                        "Debe ser < --num-shards.")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    args = parse_args()

    # ── Resolver output dir por defecto según modo ────────────────────────────
    if args.output is None:
        args.output = OUTPUT_PATH_SPATIAL if args.spatial else OUTPUT_PATH

    # ── Advertencia de batch_size para modo espacial ──────────────────────────
    if args.spatial and args.batch_size > 16:
        logging.warning(
            "--spatial con batch-size=%d puede causar OOM. "
            "Recomendado: --batch-size 8 (RTX 3080/3090) o 16 (A100).",
            args.batch_size,
        )

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

    # ── Sharding ──────────────────────────────────────────────────────────────
    if args.num_shards > 1:
        if not (0 <= args.shard_id < args.num_shards):
            raise ValueError(
                f"--shard-id {args.shard_id} fuera de rango "
                f"[0, {args.num_shards - 1}]"
            )
        # Reparto interleaved: shard 0 → [0, N, 2N, …], shard 1 → [1, N+1, 2N+1, …]
        # Garantiza distribución uniforme por categoría (los items están ordenados).
        items = items[args.shard_id :: args.num_shards]
        logging.info(
            "Shard %d/%d → %d videos asignados",
            args.shard_id, args.num_shards, len(items),
        )

    if args.limit:
        items = items[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mode_str  = "ESPACIAL (8,14,14,768) float16" if args.spatial else "ESTÁNDAR (768,) float32"
    shard_str = (f"shard {args.shard_id}/{args.num_shards}"
                 if args.num_shards > 1 else "sin sharding")
    logging.info("=" * 72)
    logging.info("VideoMAE Feature Extractor  |  UCF-Crime")
    logging.info("  Modo         : %s", mode_str)
    logging.info("  Shard        : %s", shard_str)
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
        queue, processor, model, device, args.batch_size, n,
        spatial=args.spatial,
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
