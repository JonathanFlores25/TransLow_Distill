"""
test.py
=======
Evaluation logic for VideoMAE VAD Classifier (Step3).

Design rationale vs Step2_Detector:

- evaluate() returns (auc, ap, fpr, tpr, prec, rec) — callers (mainC2FPL.py)
  receive all curve arrays needed to save .npy artefacts without needing to
  re-call sklearn functions separately.

- Clip-to-frame expansion with exact trim/pad: each clip's anomaly score is
  repeated clip_step times. The expanded array is then trimmed or padded with
  the last score to match the ground-truth frame count exactly. This mirrors
  the contract documented in FE_VideoMAE.py for frame-level reconstruction.
  Step2 used a nanlist_test index structure; Step3 uses per-video GT .npy
  files directly because FE_VideoMAE.py stores features with video stems
  in the filename, making it possible to reconstruct video groupings.

- Ground truth loading from per-video .npy files in gt_dir: Step2 loaded a
  single concatenated GT array. Step3's gt_dir contains per-video files
  named {video_stem}.npy with frame-level binary labels, consistent with
  the directory-based feature storage.

- Identical normalization stats at test time: if stats_dir is provided,
  feat_mean.npy and feat_std.npy are loaded and applied. This prevents
  test-distribution leakage that would occur if stats were recomputed on
  test features.

- Standalone __main__ with VideoMAE_VAD_Classifier: uses the SAME class name
  as train.py / model.py so saved checkpoints load correctly. Step2 used
  Model_V2 in test.py but Model_V3_Connection in main, which caused
  incompatible checkpoint keys.

- float32 throughout: no float64, no fp16 during inference.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

# SAME class name as model.py and train.py — critical for checkpoint compatibility.
from model import VideoMAE_VAD_Classifier


def _load_normalization_stats(
    stats_dir: Path,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load per-feature normalization statistics saved by mainC2FPL.py.

    Returns None if stats_dir does not contain the expected files, with a
    warning — evaluation proceeds on un-normalized features so the pipeline
    does not crash if stats were not saved yet.
    """
    mean_path = stats_dir / "feat_mean.npy"
    std_path = stats_dir / "feat_std.npy"

    if not mean_path.exists() or not std_path.exists():
        warnings.warn(
            f"Normalization stats not found in {stats_dir} "
            f"(expected feat_mean.npy and feat_std.npy). "
            f"Evaluating on un-normalized features. "
            f"Run mainC2FPL.py at least once to generate stats.",
            stacklevel=2,
        )
        return None

    return {
        "mean": np.load(mean_path).astype(np.float32),
        "std": np.load(std_path).astype(np.float32),
    }


def _collect_clip_scores(
    model: VideoMAE_VAD_Classifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, list]:
    """
    Run inference over all clips in loader.

    Returns
    -------
    scores : np.ndarray of shape (N,) float32
        Anomaly probability for each clip in loader order.
    file_paths : list of Path
        Ordered list of .npy file paths from loader.dataset.file_paths,
        used to reconstruct video groupings for frame-level expansion.
    """
    model.eval()
    all_scores: list[float] = []

    with torch.no_grad():
        for features, _labels in loader:
            features = features.to(device, dtype=torch.float32)
            probs = model(features).flatten().cpu().numpy().astype(np.float32)
            all_scores.extend(probs.tolist())

    scores = np.array(all_scores, dtype=np.float32)
    file_paths = loader.dataset.file_paths  # type: ignore[attr-defined]
    return scores, file_paths


def _expand_clip_scores_to_frames(
    scores: np.ndarray,
    file_paths: list,
    gt_dir: Path,
    clip_step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand per-clip scores to per-frame scores and concatenate GT.

    Per-clip scores from FE_VideoMAE.py cover exactly clip_step raw video
    frames (CLIP_STEP=16 in FE_VideoMAE.py). To compare with frame-level GT:
      1. Group clips by video stem (derived from filename before '_s{start}').
      2. For each video, repeat each clip's score × clip_step.
      3. Trim or pad the expanded array to match the GT frame count.
         Padding uses the last clip's score (not zero) to avoid introducing
         a spurious normal signal at the end of anomaly videos.

    Parameters
    ----------
    scores : np.ndarray (N,)
        Clip-level anomaly scores in the order they appear in the loader.
    file_paths : list of Path
        Ordered .npy file paths — same order as scores.
    gt_dir : Path
        Directory containing per-video {video_stem}.npy arrays of frame-level
        binary labels (0=normal, 1=anomaly).
    clip_step : int
        Frames per clip (= CLIP_STEP in FE_VideoMAE.py, default 16).

    Returns
    -------
    all_frame_scores : np.ndarray (M,) float32
        Concatenated frame-level scores across all videos.
    all_frame_gt : np.ndarray (M,) int32
        Concatenated frame-level ground-truth labels.
    """
    # Group clips by video stem.
    # FE_VideoMAE.py naming: {video_stem}_s{start_frame:06d}.npy
    # We split at the last occurrence of '_s' followed by 6 digits.
    from collections import OrderedDict
    import re

    _CLIP_STEM_RE = re.compile(r"^(.+)_s\d{6}$")

    # Ordered dict: video_stem → list of (clip_index, clip_start_frame)
    video_clips: OrderedDict = OrderedDict()
    for i, fp in enumerate(file_paths):
        m = _CLIP_STEM_RE.match(fp.stem)
        if m:
            video_stem = m.group(1)
        else:
            # Fallback: treat the whole stem as video stem (no start token).
            video_stem = fp.stem
        if video_stem not in video_clips:
            video_clips[video_stem] = []
        video_clips[video_stem].append(i)

    all_frame_scores: list[np.ndarray] = []
    all_frame_gt: list[np.ndarray] = []
    skipped = 0

    for video_stem, clip_indices in video_clips.items():
        # Load per-video GT. Try exact match first, then recursive glob.
        gt_path = gt_dir / f"{video_stem}.npy"
        if not gt_path.exists():
            # GT files may be in category subdirectories.
            candidates = list(gt_dir.rglob(f"{video_stem}.npy"))
            if not candidates:
                warnings.warn(
                    f"GT not found for video '{video_stem}' in {gt_dir}. "
                    f"Skipping this video in AUC computation.",
                    stacklevel=2,
                )
                skipped += 1
                continue
            gt_path = candidates[0]

        gt_frames = np.load(gt_path).astype(np.int32)
        n_frames_gt = len(gt_frames)

        # Expand clip scores to frames.
        clip_scores_video = scores[clip_indices]  # (n_clips,)
        # Each clip covers exactly clip_step raw frames.
        expanded = np.repeat(clip_scores_video, clip_step).astype(np.float32)

        # Trim or pad to match ground-truth length.
        if len(expanded) >= n_frames_gt:
            expanded = expanded[:n_frames_gt]
        else:
            # Pad with last score — preserves anomaly signal at clip boundaries
            # rather than inserting spurious zeros at video end.
            pad_len = n_frames_gt - len(expanded)
            last_score = expanded[-1] if len(expanded) > 0 else 0.0
            expanded = np.concatenate(
                [expanded, np.full(pad_len, last_score, dtype=np.float32)]
            )

        all_frame_scores.append(expanded)
        all_frame_gt.append(gt_frames)

    if skipped > 0:
        warnings.warn(
            f"Skipped {skipped} videos due to missing GT files. "
            f"AUC is computed over {len(all_frame_scores)} videos only.",
            stacklevel=2,
        )

    if len(all_frame_scores) == 0:
        raise RuntimeError(
            f"No videos could be evaluated — no GT files found in {gt_dir}. "
            f"Check that gt_dir contains per-video {'{stem}'}.npy files."
        )

    concat_scores = np.concatenate(all_frame_scores).astype(np.float32)
    concat_gt = np.concatenate(all_frame_gt).astype(np.int32)
    return concat_scores, concat_gt


def evaluate(
    model: VideoMAE_VAD_Classifier,
    loader: DataLoader,
    gt_dir: Path,
    clip_step: int,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the model on a split, returning frame-level AUC and AP.

    Parameters
    ----------
    model : VideoMAE_VAD_Classifier
        Classifier in eval() mode (this function calls model.eval()).
    loader : DataLoader
        Evaluation DataLoader. Dataset must have a .file_paths attribute
        (VideoMAEFeatureDataset). Labels from loader are NOT used — GT is
        loaded from gt_dir to enable exact frame-level evaluation.
    gt_dir : Path
        Directory with per-video frame-level GT .npy files.
    clip_step : int
        Raw frames per clip (= CLIP_STEP in FE_VideoMAE.py, default 16).
    device : torch.device
        Inference device.

    Returns
    -------
    auc_roc : float
        Area under ROC curve (frame-level).
    avg_precision : float
        Average Precision = area under PR curve (frame-level).
    fpr : np.ndarray
        False positive rates for the ROC curve.
    tpr : np.ndarray
        True positive rates for the ROC curve.
    precision : np.ndarray
        Precision values for the PR curve.
    recall : np.ndarray
        Recall values for the PR curve.
    """
    scores, file_paths = _collect_clip_scores(model, loader, device)

    frame_scores, frame_gt = _expand_clip_scores_to_frames(
        scores, file_paths, Path(gt_dir), clip_step
    )

    # ROC curve and AUC
    fpr, tpr, _thresholds_roc = roc_curve(frame_gt, frame_scores)
    auc_roc = float(auc(fpr, tpr))

    # Precision-Recall curve and Average Precision
    precision, recall, _thresholds_pr = precision_recall_curve(frame_gt, frame_scores)
    avg_precision = float(auc(recall, precision))

    return auc_roc, avg_precision, fpr, tpr, precision, recall


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    from pathlib import Path

    import torch
    from torch.utils.data import DataLoader

    from dataset import VideoMAEFeatureDataset, collate_fn
    from gpu_utils import get_optimal_device

    p = argparse.ArgumentParser(
        description=(
            "Standalone evaluation for VideoMAE VAD Classifier. "
            "Loads a checkpoint, runs inference on a split, saves "
            "FPR/TPR/precision/recall .npy files and prints AUC/AP."
        )
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to .pt checkpoint saved by mainC2FPL.py.",
    )
    p.add_argument(
        "--features_dir",
        type=Path,
        required=True,
        help="Root features directory (same as used during training).",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate (default: test).",
    )
    p.add_argument(
        "--gt_dir",
        type=Path,
        required=True,
        help="Directory with per-video frame-level GT .npy files.",
    )
    p.add_argument(
        "--stats_dir",
        type=Path,
        default=None,
        help=(
            "Directory with feat_mean.npy and feat_std.npy. "
            "If omitted, features are not normalised (not recommended)."
        ),
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where FPR/TPR/precision/recall .npy will be saved.",
    )
    p.add_argument(
        "--clip_step", type=int, default=16,
        help="Frames per clip (must match FE_VideoMAE.py CLIP_STEP).",
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)

    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = get_optimal_device()
    print(f"[test.py] Device: {device}")

    # Load normalization stats if available.
    stats = None
    if args.stats_dir is not None:
        stats = _load_normalization_stats(args.stats_dir)

    # Build dataset — uses VideoMAEFeatureDataset with optional stats.
    dataset = VideoMAEFeatureDataset(
        root_dir=args.features_dir,
        split=args.split,
        stats=stats,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_fn,
    )

    # Load model checkpoint.
    # VideoMAE_VAD_Classifier is the SAME class as used in train.py —
    # this is the fix vs Step2 where train and test used different classes.
    model = VideoMAE_VAD_Classifier(feature_dim=768)
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Support both raw state_dict and the full checkpoint dict saved by
    # mainC2FPL.py (which stores model_state, optimizer_state, epoch, val_auc).
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        saved_epoch = ckpt.get("epoch", "?")
        saved_val_auc = ckpt.get("val_auc", float("nan"))
        print(f"[test.py] Checkpoint: epoch={saved_epoch}, val_auc={saved_val_auc:.4f}")
    else:
        state_dict = ckpt

    # Strip 'module.' prefix from DataParallel-wrapped checkpoints.
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)

    auc_roc, avg_precision, fpr, tpr, precision, recall = evaluate(
        model, loader, args.gt_dir, args.clip_step, device
    )

    print(f"[test.py] AUC-ROC : {auc_roc:.4f}")
    print(f"[test.py] Avg Prec: {avg_precision:.4f}")

    # Save evaluation curves as .npy for downstream plotting / analysis.
    np.save(args.output_dir / "fpr.npy", fpr)
    np.save(args.output_dir / "tpr.npy", tpr)
    np.save(args.output_dir / "precision.npy", precision)
    np.save(args.output_dir / "recall.npy", recall)

    print(f"[test.py] Curves saved to {args.output_dir}")
    sys.exit(0)
