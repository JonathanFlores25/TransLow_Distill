"""
viz_v2.py
=========
Visualización por video para VideoMAE VAD Classifier v2.

Genera una imagen por video mostrando:
  · Score del modelo (línea azul, frame-level)
  · GT temporal (segmentos sombreados en verde)
  · Líneas verticales en los bordes del GT
  · Score GT como línea de referencia (binaria 0/1 suavizada)
  · Estadísticas: AUC y AP individuales del video en el título

Basado en plot_video() de src/inference_arconte.py, adaptado para no
depender de is_active (VideoMAE no genera ON/OFF per clip como los expertos).

Uso típico desde mainVideoMAE_v2.py
-------------------------------------
    from viz_v2 import save_epoch_viz

    save_epoch_viz(
        model        = model,
        val_loader   = val_loader,
        annotations  = annotations,   # dict[str, VideoAnnotation]
        output_dir   = args.output_dir / "viz",
        epoch        = epoch,
        device       = device,
        clip_step    = args.clip_step,
    )
"""

from __future__ import annotations

import sys
import os
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

# Importar desde el repo raíz
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.metrics_library import (
    build_gt_frame_array,
    clips_to_frames,
    frame_auc,
    frame_ap,
    parse_gt_annotations,
)


# ─────────────────────────────────────────────────────────────────────────────
# Plot de un solo video
# ─────────────────────────────────────────────────────────────────────────────

def plot_video_scores(
    stem:         str,
    clip_starts:  list[int],
    clip_scores:  np.ndarray,
    annotation,                   # VideoAnnotation de metrics_library
    out_path:     Path,
    clip_step:    int  = 16,
    epoch:        Optional[int] = None,
) -> None:
    """
    Genera y guarda el plot de scores frame-level para un video.

    Parámetros
    ----------
    stem        : nombre del video sin extensión (e.g. "Assault028_x264")
    clip_starts : lista de frame de inicio de cada clip
    clip_scores : array (N,) de scores del clasificador ∈ [0,1]
    annotation  : VideoAnnotation con segments y category
    out_path    : ruta de salida del PNG
    clip_step   : frames por clip (default 16)
    epoch       : número de época para el título (opcional)
    """
    if len(clip_starts) == 0:
        return

    total_frames = max(clip_starts) + clip_step
    active       = np.ones(len(clip_starts), dtype=bool)

    frame_scores, _ = clips_to_frames(
        clip_starts, clip_scores, active, total_frames, clip_step
    )
    gt = build_gt_frame_array(annotation, total_frames)

    # Métricas individuales del video
    try:
        v_auc = frame_auc(frame_scores, gt)
        v_ap  = frame_ap(frame_scores, gt)
        if np.isnan(v_auc): v_auc = 0.5
        if np.isnan(v_ap):  v_ap  = 0.0
    except Exception:
        v_auc = v_ap = float("nan")

    x = np.arange(total_frames)

    fig, ax = plt.subplots(figsize=(14, 4))

    # ── GT sombreado ──────────────────────────────────────────────────────────
    for seg_idx, (seg_s, seg_e) in enumerate(annotation.segments):
        label = "Ground Truth" if seg_idx == 0 else "_nolegend_"
        ax.axvspan(
            seg_s, min(seg_e, total_frames - 1),
            alpha=0.20, color="limegreen", label=label
        )
        # Bordes del segmento GT
        ax.axvline(seg_s, color="green", lw=0.8, ls="--", alpha=0.6)
        ax.axvline(min(seg_e, total_frames - 1),
                   color="green", lw=0.8, ls="--", alpha=0.6)

    # ── GT binario como referencia (step) ─────────────────────────────────────
    ax.step(x, gt.astype(np.float32) * 0.92,
            color="limegreen", lw=0.7, alpha=0.5,
            where="post", label="GT binario", zorder=2)

    # ── Score del modelo ──────────────────────────────────────────────────────
    ax.plot(x, frame_scores, color="#1f77b4", lw=1.4,
            label="VideoMAE Score", zorder=3)

    # ── Leyenda ───────────────────────────────────────────────────────────────
    gt_patch   = mpatches.Patch(color="limegreen", alpha=0.35, label="Ground Truth")
    score_line = plt.Line2D([0], [0], color="#1f77b4", lw=1.5, label="VideoMAE Score")
    gt_line    = plt.Line2D([0], [0], color="limegreen", lw=0.8, alpha=0.6,
                             label="GT binario (ref)")
    ax.legend(handles=[gt_patch, score_line, gt_line],
              loc="upper right", fontsize=9, framealpha=0.75)

    # ── Estadísticas ──────────────────────────────────────────────────────────
    peak = float(clip_scores.max()) if len(clip_scores) else 0.0
    mean = float(clip_scores.mean()) if len(clip_scores) else 0.0
    ax.text(
        0.01, 0.97,
        f"AUC={v_auc:.3f}  AP={v_ap:.3f}  peak={peak:.3f}  mean={mean:.3f}",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=8.5, color="#2C3E50",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    # ── Título ────────────────────────────────────────────────────────────────
    cat     = getattr(annotation, "category", "?")
    ep_str  = f"  [epoch {epoch}]" if epoch is not None else ""
    ax.set_title(
        f"{stem}   GT: {cat}{ep_str}",
        fontsize=11, fontweight="bold",
    )

    ax.set_xlim(0, total_frames)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Frame index", fontsize=10)
    ax.set_ylabel("Anomaly Score [0–1]", fontsize=10)
    ax.grid(axis="y", ls="--", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Visualización de una época completa (val set)
# ─────────────────────────────────────────────────────────────────────────────

def save_epoch_viz(
    model,
    val_loader,
    gt_annotation_path: Path,
    output_dir:         Path,
    epoch:              int,
    device:             torch.device,
    clip_step:          int = 16,
    max_videos:         Optional[int] = None,
) -> None:
    """
    Genera plots frame-level para todos los videos del val set.

    Los PNGs se guardan en:
        output_dir/epoch_{epoch:03d}/{Category}/{stem}.png

    Parámetros
    ----------
    model               : VideoMAE_VAD_Classifier_v2 (o wrapper _PredictWrapper)
    val_loader          : DataLoader del val set (Dataset_VideoMAE, test_Mode="Validacion")
    gt_annotation_path  : ruta a Temporal_Anomaly_Annotation.txt
    output_dir          : directorio raíz de visualizaciones
    epoch               : número de época actual
    device              : dispositivo torch
    clip_step           : frames por clip (default 16)
    max_videos          : limitar a N videos (None = todos)
    """
    if not Path(gt_annotation_path).exists():
        warnings.warn(f"[viz_v2] gt_annotation no encontrado: {gt_annotation_path}",
                      stacklevel=2)
        return

    annotations = parse_gt_annotations(gt_annotation_path)
    dataset     = val_loader.dataset

    # ── Inferencia completa del val set ───────────────────────────────────────
    model.eval()
    pred_parts: list[np.ndarray] = []
    with torch.no_grad():
        for features in val_loader:
            features = features.to(device, dtype=torch.float32)
            scores   = model(features)               # predict() o forward()[0]
            # Soporta tanto _PredictWrapper (devuelve tensor) como modelo directo
            if isinstance(scores, tuple):
                scores = scores[0]
            pred_parts.append(scores.flatten().cpu().numpy())

    clip_scores_all = np.concatenate(pred_parts, axis=0)

    # ── Agrupar clips por video ───────────────────────────────────────────────
    video_clip_idx: dict[str, list[int]] = defaultdict(list)
    for idx, stem in enumerate(dataset.video_stems):
        video_clip_idx[stem].append(idx)

    # ── Generar un plot por video ─────────────────────────────────────────────
    epoch_dir = Path(output_dir) / f"epoch_{epoch:03d}"
    saved = skipped = 0

    stems_sorted = sorted(video_clip_idx.keys())
    if max_videos:
        stems_sorted = stems_sorted[:max_videos]

    for stem in stems_sorted:
        ann = annotations.get(stem)
        if ann is None:
            skipped += 1
            continue

        indices     = video_clip_idx[stem]
        starts      = [dataset.clip_starts[i] for i in indices]
        scores_v    = clip_scores_all[indices]
        category    = dataset.categories[indices[0]] if indices else "unknown"

        out_path = epoch_dir / category / f"{stem}.png"

        try:
            plot_video_scores(
                stem        = stem,
                clip_starts = starts,
                clip_scores = scores_v,
                annotation  = ann,
                out_path    = out_path,
                clip_step   = clip_step,
                epoch       = epoch,
            )
            saved += 1
        except Exception as exc:
            warnings.warn(f"[viz_v2] Plot falló para {stem}: {exc}", stacklevel=2)
            skipped += 1

    print(f"[viz_v2] epoch {epoch}: {saved} plots guardados en {epoch_dir}  "
          f"({skipped} omitidos)")
