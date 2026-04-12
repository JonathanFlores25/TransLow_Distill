"""
option_v2.py
============
Argumentos CLI para VideoMAE VAD Classifier v2 (Metric Learning).

Extiende los argumentos de v1 con:
  · Arquitectura  : proj_hidden, proj_out
  · Metric loss   : triplet_weight, tau
  · Regularización: label_smoothing
  · Scheduler     : warmup_epochs, eta_min
"""

import argparse
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "VideoMAE VAD Classifier v2 — Metric Learning Edition. "
            "Entrena un clasificador con Encoder compartido + "
            "Classification Head (BCE) + Projection Head (Soft Triplet OHNM)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data paths ────────────────────────────────────────────────────────────
    p.add_argument("--features_dir",     type=Path, required=True)
    p.add_argument("--clip_names_path",  type=Path, default=None)
    p.add_argument("--gt_binary_path",   type=Path, default=None)
    p.add_argument("--pseudo_threshold", type=float, default=0.5)
    p.add_argument("--gt_annotation",    type=Path, default=None)
    p.add_argument("--test_cache_dir",   type=Path, default=None)
    p.add_argument("--test_video_root",  type=Path, default=None)
    p.add_argument("--ckpt_dir",         type=Path, required=True)
    p.add_argument("--output_dir",       type=Path, required=True)
    p.add_argument("--stats_dir",        type=Path, required=True)

    # ── VideoMAE / feature constants ──────────────────────────────────────────
    p.add_argument("--feature_dim",  type=int, default=768)
    p.add_argument("--clip_step",    type=int, default=16)
    p.add_argument("--tubelet_size", type=int, default=2)
    p.add_argument("--num_tokens",   type=int, default=1568)

    # ── Arquitectura del Projection Head ─────────────────────────────────────
    p.add_argument(
        "--proj_hidden",
        type=int,
        default=256,
        help="Dimensión oculta del Projection Head (512 → proj_hidden → proj_out).",
    )
    p.add_argument(
        "--proj_out",
        type=int,
        default=128,
        help="Dimensión del embedding de salida del Projection Head. "
             "128 es el estándar en SimCLR/MoCo para features de esta escala.",
    )

    # ── Training hyperparameters ──────────────────────────────────────────────
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--max_epoch",   type=int,   default=100, dest="max_epoch")
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=5e-4)
    p.add_argument("--num_workers", type=int,   default=4)

    # ── Metric Loss ───────────────────────────────────────────────────────────
    p.add_argument(
        "--triplet_weight",
        type=float,
        default=0.3,
        help="λ_t: peso del Soft Triplet en L_total = L_bce + λ_t·L_triplet. "
             "0.0 desactiva el Soft Triplet (equivale a v1). "
             "Rango recomendado: [0.1, 0.5].",
    )
    p.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="Temperatura del Soft Triplet: log(1+exp((d_ap²-d_an²)/τ)). "
             "τ bajo (0.1-0.3) → gradiente concentrado en hard cases. "
             "τ alto (0.7-1.0) → gradiente distribuido, más robusto a ruido. "
             "Con pseudo-labels UCF-Crime, τ=0.5 es un buen punto de partida.",
    )

    # ── Label Smoothing ───────────────────────────────────────────────────────
    p.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Suavizado de etiquetas para BCE. "
             "target = label*(1-ε) + ε/2. "
             "0.0 desactiva el suavizado.",
    )

    # ── LR Scheduler: CosineAnnealing + warmup ────────────────────────────────
    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Épocas de warmup lineal (lr: lr*0.1 → lr). "
             "Después del warmup arranca CosineAnnealingLR. "
             "0 desactiva el warmup.",
    )
    p.add_argument(
        "--eta_min",
        type=float,
        default=1e-6,
        help="LR mínimo al final del CosineAnnealing.",
    )

    return p


def parse_args() -> argparse.Namespace:
    return get_parser().parse_args()
