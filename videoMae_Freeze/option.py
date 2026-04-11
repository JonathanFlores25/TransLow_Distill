"""
option.py
=========
Pure argparse configuration for VideoMAE VAD Classifier (Step3).

Design rationale vs Step2_Detector:
- No external config module import: Step2 imported a project-level `config`
  module and did parse_args() at module level — both cause import-time side
  effects that break multi-process DataLoader workers. Here parse_args() is
  a plain function, called only inside if __name__ == '__main__' guards.
- feature_dim defaults to 768: VideoMAE ViT-Base hidden_size is fixed at 768
  (12 layers × 12 heads × 64 head_dim). Never change this without re-extracting
  features with a different backbone.
- tubelet_size / num_tokens are informational: they document the upstream
  tokenization contract (tubelet=2 → 8 temporal positions, 14×14 spatial grid
  → 196 tokens per position, 8×196=1568 total) so readers understand why the
  feature vector encodes spatiotemporal structure jointly.
- stats_dir: new in Step3, needed because VideoMAE was pre-trained on
  Kinetics-400 (clean action clips) and UCF-Crime surveillance features have
  a significant domain shift — per-feature z-score normalization computed on
  the training split only must be persisted so test.py applies identical
  statistics without recomputing from the (unseen) test distribution.
- pos_weight=-1 triggers auto-computation from training label counts at
  runtime, avoiding the need to pre-specify class ratios.
"""

import argparse
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    """
    Build and return the argument parser.
    Called explicitly by each entry-point; never at import time.
    """
    p = argparse.ArgumentParser(
        description=(
            "VideoMAE VAD Classifier — trains a lightweight head on top of "
            "frozen VideoMAE (MCG-NJU/videomae-base-finetuned-kinetics) "
            "clip-level features extracted by FE_VideoMAE.py. "
            "Input features are (768,) float32 mean-pooled patch tokens."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data paths ───────────────────────────────────────────────────────────
    p.add_argument(
        "--features_dir",
        type=Path,
        required=True,
        help=(
            "Root directory produced by FE_VideoMAE.py. Expected layout: "
            "<features_dir>/<split>/<category>/<stem>_s<start:06d>.npy  "
            "where split ∈ {train, val, test} and category ∈ {fight, crash, "
            "fire, robbery, carparts, normal}."
        ),
    )
    p.add_argument(
        "--clip_names_path",
        type=Path,
        default=None,
        help=(
            "Ruta a clip_names.npy generado por pseudo_label_generator.py. "
            "Array (N,) de strings con el nombre de cada clip del train set. "
            "Equivalente a mainConca_Path de Step2 pero para lookup de labels. "
            "Ejemplo: data/pseudo_labels/clip_names.npy"
        ),
    )
    p.add_argument(
        "--gt_binary_path",
        type=Path,
        default=None,
        help=(
            "Ruta a gt_binary.npy generado por pseudo_label_generator.py. "
            "Array (N,) float32 con soft labels 0.0-1.0 por clip de train. "
            "Equivalente a pseudofile_train de Step2. "
            "Ejemplo: data/pseudo_labels/gt_binary.npy"
        ),
    )
    p.add_argument(
        "--pseudo_threshold",
        type=float,
        default=0.5,
        help=(
            "Umbral para binarizar gt_binary: label = int(gt_binary > threshold). "
            "Default 0.5. Bajar a 0.3 incluye más clips como positivos (más recall)."
        ),
    )
    p.add_argument(
        "--gt_annotation",
        type=Path,
        default=None,
        help=(
            "Ruta a Temporal_Anomaly_Annotation.txt para evaluación frame-level. "
            "Ejemplo: resources/Temporal_Anomaly_Annotation.txt"
        ),
    )
    p.add_argument(
        "--test_cache_dir",
        type=Path,
        default=None,
        help=(
            "Directorio donde se cachean los features (.npy) del test set "
            "extraídos por test_Over_online(). Primera pasada los genera, "
            "las siguientes los cargan directo sin correr el backbone. "
            "Default: data/vmae_features_test/ junto a features_dir."
        ),
    )
    p.add_argument(
        "--test_video_root",
        type=Path,
        default=None,
        help=(
            "Raíz de los videos del test set oficial UCF-Crime. "
            "Estructura esperada: {test_video_root}/{Category}/{stem}.mp4  "
            "Ejemplo: /mnt/gpu02/datasets/Data_Arco_V2/UCF-Crime/videos  "
            "Usado por test_Over_online() para evaluación sin guardar features a disco."
        ),
    )
    p.add_argument(
        "--ckpt_dir",
        type=Path,
        required=True,
        help="Directory where model checkpoints (.pt) will be saved.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help=(
            "Directory for evaluation outputs: FPR/TPR/precision/recall .npy "
            "curves, training plot .png files, and per-epoch CSV log."
        ),
    )
    p.add_argument(
        "--stats_dir",
        type=Path,
        required=True,
        help=(
            "Directory where feature normalization statistics are persisted. "
            "train.py writes feat_mean.npy and feat_std.npy here after "
            "computing Welford online stats on the training split. "
            "test.py loads these same files so evaluation uses identical "
            "normalization without touching the test distribution."
        ),
    )

    # ── VideoMAE architecture constants (informational / assertion guards) ────
    p.add_argument(
        "--feature_dim",
        type=int,
        default=768,
        help=(
            "Hidden size of VideoMAE ViT-Base encoder. MUST be 768 for "
            "MCG-NJU/videomae-base-finetuned-kinetics. Changing this "
            "requires re-extracting features with a different backbone."
        ),
    )
    p.add_argument(
        "--tubelet_size",
        type=int,
        default=2,
        help=(
            "[Informational] Temporal depth of each VideoMAE tube embedding. "
            "With CLIP_LEN=16 frames and tubelet_size=2, the encoder sees "
            "16//2=8 temporal positions. Changing this invalidates all "
            "pre-extracted .npy features."
        ),
    )
    p.add_argument(
        "--num_tokens",
        type=int,
        default=1568,
        help=(
            "[Informational] Total patch tokens per clip in VideoMAE: "
            "(CLIP_LEN // tubelet_size) × (224 // patch_size)² = 8 × 196 = "
            "1568. FE_VideoMAE.py mean-pools these 1568 tokens into the "
            "(768,) feature vector. The 768 dims are therefore spatiotemporally "
            "entangled — not independent spatial channels."
        ),
    )
    p.add_argument(
        "--clip_step",
        type=int,
        default=16,
        help=(
            "Number of raw video frames each clip covers (= CLIP_STEP in "
            "FE_VideoMAE.py). Used in evaluate() to expand per-clip anomaly "
            "scores to per-frame scores via np.repeat(score, clip_step), "
            "then trimmed/padded to match the ground-truth frame count."
        ),
    )

    # ── Training hyperparameters ──────────────────────────────────────────────
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help=(
            "Clips per batch. BalancedBatchSampler ensures 50%% normal / "
            "50%% anomaly within each batch regardless of dataset imbalance."
        ),
    )
    p.add_argument(
        "--max_epoch", "--epochs",
        type=int,
        default=100,
        dest="max_epoch",
        help="Total epochs de entrenamiento.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate inicial (SGD).",
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay para SGD (igual que Step2).",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Workers del DataLoader para I/O de archivos .npy.",
    )

    return p


def parse_args() -> argparse.Namespace:
    """Convenience wrapper — parse sys.argv using the standard parser."""
    return get_parser().parse_args()
