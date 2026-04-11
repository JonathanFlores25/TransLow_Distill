"""
test.py
=======
Evaluation logic for VideoMAE VAD Classifier.

Dos modos de evaluación
-----------------------
test_Over(dataloader, model, args, device)
    Evaluación rápida con features pre-extraídas (.npy).
    Usado cada época para el set de validación.
    - Con args.gt_annotation: AUC/AP frame-level vía metrics_library.
    - Sin args.gt_annotation: AUC/AP clip-level con labels de directorio.

test_Over_online(backbone, processor, classifier, args, device)
    Evaluación online para el test set oficial (290 videos).
    video crudo → VideoMAE FE en línea → (768,) → clasificador → score.
    Sin guardar features a disco.
    Requiere args.gt_annotation y args.test_video_root.
"""

from __future__ import annotations

import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Generator, List, Tuple

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics_library import (
    build_gt_frame_array,
    clips_to_frames,
    frame_ap,
    frame_auc,
    parse_gt_annotations,
)

# Clip params — deben coincidir con FE_VideoMAE.py
_CLIP_LEN  = 16
_STRIDE    = 2
_OUT_SIZE  = 224


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad: leer clips de un video en memoria
# ─────────────────────────────────────────────────────────────────────────────

def _read_clips_from_video(
    video_path: Path,
    clip_len:   int = _CLIP_LEN,
    stride:     int = _STRIDE,
    clip_step:  int = 16,
) -> Generator[Tuple[int, List[np.ndarray]], None, None]:
    """
    Generator: yield (clip_start, frames_rgb) para cada clip del video.

    Usa buffer deslizante idéntico a FE_VideoMAE.py — misma segmentación
    temporal garantizada. frames_rgb es lista de clip_len arrays (H, W, 3).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warnings.warn(f"No se puede abrir: {video_path}", stacklevel=2)
        return

    frame_buffer: dict[int, np.ndarray] = {}
    next_start = 0
    frame_idx  = 0

    while True:
        ret, raw = cap.read()
        if not ret:
            break

        frame = cv2.resize(raw, (_OUT_SIZE, _OUT_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer[frame_idx] = frame
        frame_idx += 1

        while True:
            clip_end = next_start + (clip_len - 1) * stride
            if clip_end >= frame_idx:
                break
            src_idx = [next_start + i * stride for i in range(clip_len)]
            yield next_start, [frame_buffer[i] for i in src_idx]
            next_start += clip_step
            for k in [k for k in frame_buffer if k < next_start]:
                del frame_buffer[k]

    cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación rápida — features pre-extraídas (val set)
# ─────────────────────────────────────────────────────────────────────────────

def test_Over(dataloader, model, args, device) -> Tuple[float, float]:
    """
    Evaluación con features .npy pre-extraídas. Devuelve (auc, ap).
    Usado para el set de validación cada época (rápido).
    """
    model.eval()
    dataset   = dataloader.dataset
    clip_step = getattr(args, "clip_step", 16)

    pred_parts: list[np.ndarray] = []
    with torch.no_grad():
        for features in tqdm(dataloader, desc="Val inference", leave=False):
            features = features.to(device, dtype=torch.float32)
            scores   = model(features).flatten().detach().cpu().numpy().astype(np.float32)
            pred_parts.append(scores)

    clip_scores = np.concatenate(pred_parts, axis=0)

    gt_path = getattr(args, "gt_annotation", None)
    if gt_path is not None and Path(gt_path).exists():
        auc_val, ap_val = _frame_level_metrics(clip_scores, dataset, gt_path, clip_step)
    else:
        auc_val, ap_val = _clip_level_metrics(clip_scores, dataset)

    print(f"[Val]  AUC={auc_val:.4f}  AP={ap_val:.4f}")
    return auc_val, ap_val


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación online — video crudo → FE → clasificador (test set oficial)
# ─────────────────────────────────────────────────────────────────────────────

def test_Over_online(backbone, processor, classifier, args, device) -> Tuple[float, float]:
    """
    Evaluación online sin features pre-extraídas a disco.

    video crudo → VideoMAE backbone → (768,) → clasificador → score
    Agrupa por video, expande a frames, compara con GT temporal.

    Parámetros requeridos en args
    -----------------------------
    args.gt_annotation    — Temporal_Anomaly_Annotation.txt
    args.test_video_root  — raíz de videos UCF-Crime ({Category}/{stem}.mp4)
    args.clip_step        — 16 (por defecto)
    args.batch_size       — clips por forward pass del backbone
    """
    gt_path    = Path(args.gt_annotation)
    video_root = Path(args.test_video_root)
    clip_step  = getattr(args, "clip_step", 16)
    batch_size = getattr(args, "batch_size", 32)

    annotations = parse_gt_annotations(gt_path)

    # Parsear anotaciones para obtener ruta de cada video
    # Formato: VideoName.mp4  Category  Start1 End1 [Start2 End2]
    video_entries: list[tuple[Path, str]] = []
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            video_filename = parts[0]          # e.g. "Abuse028_x264.mp4"
            category       = parts[1]          # e.g. "Abuse"
            video_path     = video_root / category / video_filename
            video_stem     = Path(video_filename).stem
            video_entries.append((video_path, video_stem))

    # Directorio de caché para features del test set
    cache_dir = Path(getattr(args, "test_cache_dir", None) or
                     Path(args.features_dir).parent / "vmae_features_test")

    backbone.eval()
    classifier.eval()

    all_frame_scores: list[np.ndarray] = []
    all_frame_gt:     list[np.ndarray] = []
    skipped  = 0
    cached   = 0
    computed = 0

    for video_path, stem in tqdm(video_entries, desc="Test online", leave=True):
        ann = annotations.get(stem)
        if ann is None:
            skipped += 1
            continue

        if not video_path.exists():
            warnings.warn(f"Video no encontrado: {video_path}", stacklevel=2)
            skipped += 1
            continue

        # Directorio de caché para este video
        video_cache_dir = cache_dir / video_path.parent.name   # {Category}/
        video_cache_dir.mkdir(parents=True, exist_ok=True)

        # Descubrir clips del video (necesitamos los starts siempre)
        clip_starts_v: list[int]              = []
        clip_frames_v: list[list[np.ndarray]] = []  # vacío si todo está cacheado

        for start, frames in _read_clips_from_video(video_path, clip_step=clip_step):
            cache_path = video_cache_dir / f"{stem}_s{start:06d}.npy"
            if cache_path.exists():
                clip_starts_v.append(start)
                clip_frames_v.append(None)   # marcador: cargar de disco
            else:
                clip_starts_v.append(start)
                clip_frames_v.append(frames)

        if not clip_starts_v:
            skipped += 1
            continue

        # Forward VideoMAE solo para clips sin caché, luego clasificador
        clip_embeddings: list[np.ndarray] = [None] * len(clip_starts_v)

        # Cargar los ya cacheados
        for i, (start, frames) in enumerate(zip(clip_starts_v, clip_frames_v)):
            if frames is None:
                cache_path = video_cache_dir / f"{stem}_s{start:06d}.npy"
                clip_embeddings[i] = np.load(cache_path).astype(np.float32)
                cached += 1

        # Procesar con backbone los que faltan (en batches)
        pending_idx    = [i for i, f in enumerate(clip_frames_v) if f is not None]
        pending_frames = [clip_frames_v[i] for i in pending_idx]

        with torch.no_grad():
            for b in range(0, len(pending_frames), batch_size):
                batch_frames = pending_frames[b : b + batch_size]
                batch_idx    = pending_idx[b : b + batch_size]

                inputs = processor(batch_frames, return_tensors="pt")
                inputs = {
                    k: v.to(device, dtype=torch.float16 if v.dtype == torch.float32 else v.dtype)
                    for k, v in inputs.items()
                }
                emb = backbone(**inputs).last_hidden_state.mean(dim=1).float().cpu().numpy()

                for j, idx in enumerate(batch_idx):
                    start      = clip_starts_v[idx]
                    cache_path = video_cache_dir / f"{stem}_s{start:06d}.npy"
                    np.save(cache_path, emb[j])
                    clip_embeddings[idx] = emb[j]
                    computed += 1

        # Clasificador sobre todos los embeddings del video
        emb_tensor = torch.tensor(np.stack(clip_embeddings), dtype=torch.float32).to(device)
        with torch.no_grad():
            clip_scores_v = classifier(emb_tensor).flatten().cpu().numpy().tolist()

        clip_scores_arr = np.array(clip_scores_v, dtype=np.float32)
        active          = np.ones(len(clip_scores_arr), dtype=bool)
        total_frames    = max(clip_starts_v) + clip_step

        frame_scores, _ = clips_to_frames(
            clip_starts_v, clip_scores_arr, active, total_frames, clip_step
        )
        gt = build_gt_frame_array(ann, total_frames)

        all_frame_scores.append(frame_scores)
        all_frame_gt.append(gt)

    print(f"[Test] clips: {cached} de caché | {computed} extraídos con backbone | {skipped} videos omitidos")

    if not all_frame_scores:
        warnings.warn("[test_Over_online] Sin datos para evaluar. Retorna (0.5, 0.0).", stacklevel=2)
        return 0.5, 0.0

    flat_scores = np.concatenate(all_frame_scores).astype(np.float32)
    flat_gt     = np.concatenate(all_frame_gt).astype(np.int32)

    auc_val = frame_auc(flat_scores, flat_gt)
    ap_val  = frame_ap(flat_scores,  flat_gt)

    if np.isnan(auc_val): auc_val = 0.5
    if np.isnan(ap_val):  ap_val  = 0.0

    print(f"[Test] AUC={auc_val:.4f}  AP={ap_val:.4f}")
    return float(auc_val), float(ap_val)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _frame_level_metrics(clip_scores, dataset, gt_annotation_path, clip_step):
    annotations = parse_gt_annotations(gt_annotation_path)
    video_clip_idx: dict[str, list[int]] = defaultdict(list)
    for idx, stem in enumerate(dataset.video_stems):
        video_clip_idx[stem].append(idx)

    all_frame_scores, all_frame_gt = [], []
    skipped = 0
    for stem in sorted(video_clip_idx):
        ann = annotations.get(stem)
        if ann is None:
            skipped += 1
            continue
        indices = video_clip_idx[stem]
        starts  = [dataset.clip_starts[i] for i in indices]
        scores  = clip_scores[indices]
        active  = np.ones(len(indices), dtype=bool)
        total_frames = max(starts) + clip_step
        frame_scores, _ = clips_to_frames(starts, scores, active, total_frames, clip_step)
        gt = build_gt_frame_array(ann, total_frames)
        all_frame_scores.append(frame_scores)
        all_frame_gt.append(gt)

    if not all_frame_scores:
        return 0.5, 0.0

    flat_scores = np.concatenate(all_frame_scores).astype(np.float32)
    flat_gt     = np.concatenate(all_frame_gt).astype(np.int32)
    auc_val = frame_auc(flat_scores, flat_gt)
    ap_val  = frame_ap(flat_scores,  flat_gt)
    if np.isnan(auc_val): auc_val = 0.5
    if np.isnan(ap_val):  ap_val  = 0.0
    return float(auc_val), float(ap_val)


def _clip_level_metrics(clip_scores, dataset):
    gt_clips = np.array(dataset.labels_all, dtype=np.int32)
    n = min(len(clip_scores), len(gt_clips))
    clip_scores, gt_clips = clip_scores[:n], gt_clips[:n]
    if len(np.unique(gt_clips)) < 2:
        return 0.5, 0.0
    return (
        float(roc_auc_score(gt_clips, clip_scores)),
        float(average_precision_score(gt_clips, clip_scores)),
    )
