"""
src/metrics_library.py
======================
Model-agnostic metrics library for Video Anomaly Detection (VAD).
Compatible with Arconte, VideoMAE, I3D, or any backbone that produces
clip-level scores and is_active flags.

API contract
------------
All public functions receive plain numpy arrays or Python dicts —
no framework-specific objects. This ensures the library can be reused
across training, fine-tuning, and production evaluation pipelines.

Score convention
----------------
Raw scores from CLIP-based experts range roughly in (-∞, +∞). Before
feeding into any metric, call apply_relu_scale() once to map them to
[0, 1] using the global saturation factor S=5.0. Do NOT normalize
per-video: global scaling preserves cross-camera comparability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# ── Constants matching Arconte data pipeline ──────────────────────────────────
CLIP_LEN  = 16    # frames sampled per clip (sparse sampling with STRIDE)
STRIDE    = 2     # temporal stride between sampled frames
CLIP_STEP = 16    # consecutive raw frames "owned" by each clip
SAT_SCALE = 5.0   # S: raw score saturates to 1.0 at this value

# ── Full UCF-Crime category → Arconte expert mapping ─────────────────────────
CATEGORY_MAP: Dict[str, str] = {
    # Direct Arconte categories
    "Fighting":      "fight",
    "Assault":       "fight",
    "RoadAccidents": "crash",
    "Arson":         "fire",
    "Explosion":     "fire",
    "Burglary":      "robbery",
    "Robbery":       "robbery",
    "Stealing":      "carparts",
    "Normal":        "normal",
    # Additional UCF-Crime categories (mapped to closest expert)
    "Abuse":         "fight",
    "Arrest":        "fight",
    "Shooting":      "fight",
    "Shoplifting":   "carparts",
    "Vandalism":     "robbery",
    "Accident":      "crash",
}


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VideoAnnotation:
    """Ground-truth annotation for one video from Temporal_Anomaly_Annotation.txt."""
    video_name: str                    # stem without extension, e.g. "Arson007_x264"
    category:   str                    # raw UCF-Crime category, e.g. "Arson"
    expert:     str                    # mapped Arconte expert, e.g. "fire"
    is_anomaly: bool                   # False for Normal videos
    segments:   List[Tuple[int, int]]  # [(start_frame, end_frame), ...] — at most 2


@dataclass
class VideoResult:
    """
    Expert-scoring result for one video (from expert_scores.npy).

    clips: list of dicts with keys:
        clip_name : str   — "{stem}_s{start:06d}"
        start     : int   — first raw frame of the clip
        is_active : bool  — True if expert fired during this clip
        score     : float — raw CLIP score (un-normalized)
    """
    video_name:  str
    expert_type: str
    clips: List[Dict[str, Any]]

    @property
    def is_active_video(self) -> bool:
        """Video-level ON/OFF: True if any clip fired (production logic)."""
        return any(c["is_active"] for c in self.clips)


@dataclass
class FrameArrays:
    """Flat, frame-level arrays ready for metric computation across all videos."""
    scores:     np.ndarray   # float32 (N,) — normalized [0, 1]
    gt_binary:  np.ndarray   # int32   (N,) — 1=anomaly, 0=normal
    is_active:  np.ndarray   # bool    (N,) — ON/OFF alert state per frame
    video_ids:  np.ndarray   # int32   (N,) — video index (for per-video slicing)


@dataclass
class MetricsReport:
    """All computed VAD metrics in one place."""
    # Frame-level (requires score continuity)
    frame_auc: float = 0.0   # AUC-ROC at frame level
    frame_ap:  float = 0.0   # Average Precision (PR-AUC) at frame level
    # Video-level (binary ON/OFF from is_active)
    acc:  float = 0.0        # Overall accuracy
    maa:  float = 0.0        # Mean Average Accuracy (per-class normalized)
    # Abnormal-only temporal localization
    abn_auc: float = 0.0     # AUC within anomalous videos only
    abn_ap:  float = 0.0     # AP within anomalous videos only
    # Hardware benchmarks (populated separately via benchmark_hardware)
    fps:     float = 0.0
    gflops:  float = 0.0
    vram_gb: float = 0.0
    # Per-class breakdown
    per_class_acc: Dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Ground-truth parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_gt_annotations(txt_path: str | Path) -> Dict[str, VideoAnnotation]:
    """
    Parse Temporal_Anomaly_Annotation.txt.

    Line format (whitespace-separated):
        VideoName.mp4  Category  Start1  End1  [Start2  End2]

    For Normal videos Start/End are -1. For anomalies, frame numbers
    are absolute indices into the raw video.

    Returns
    -------
    dict  video_stem → VideoAnnotation
          key is the stem without extension, e.g. "Arson007_x264"
    """
    annotations: Dict[str, VideoAnnotation] = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue

            video_name = Path(parts[0]).stem
            category   = parts[1]
            is_anomaly = (category != "Normal")
            expert     = CATEGORY_MAP.get(category, "normal")

            segments: List[Tuple[int, int]] = []
            for seg_idx in range(2):
                col = 2 + seg_idx * 2
                if col + 1 >= len(parts):
                    break
                s, e = int(parts[col]), int(parts[col + 1])
                if s != -1 and e != -1:
                    segments.append((s, e))

            annotations[video_name] = VideoAnnotation(
                video_name = video_name,
                category   = category,
                expert     = expert,
                is_anomaly = is_anomaly,
                segments   = segments,
            )
    return annotations


# ──────────────────────────────────────────────────────────────────────────────
# 2. Score normalization (global — not per-video)
# ──────────────────────────────────────────────────────────────────────────────

def apply_relu_scale(
    scores:    np.ndarray,
    sat_scale: float = SAT_SCALE,
) -> np.ndarray:
    """
    Map raw expert scores to [0, 1] using global ReLU + saturation.

    Formula: score_norm = clip(max(0, x) / S, 0, 1)

    Rationale: scores below 0 carry no anomaly information (ReLU discards
    them). Dividing by S=5.0 aligns with the typical range of CLIP cosine
    similarity differences. Saturation at 1.0 caps outliers.

    This must NOT be applied per-video so that relative magnitudes are
    preserved across cameras (e.g. a weak fire signal stays weak even if
    it's the largest score in that video).

    Parameters
    ----------
    scores    : raw float scores from expert.predict() — can be negative
    sat_scale : S — raw score value that saturates to 1.0 (default 5.0)

    Returns
    -------
    float32 ndarray with values in [0, 1]
    """
    scores = np.asarray(scores, dtype=np.float32)
    return np.clip(np.maximum(0.0, scores) / sat_scale, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Clip → Frame expansion
# ──────────────────────────────────────────────────────────────────────────────

def clips_to_frames(
    clip_starts:  List[int],
    clip_scores:  np.ndarray,
    clip_active:  np.ndarray,
    total_frames: int,
    clip_step:    int = CLIP_STEP,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Broadcast clip-level scores and is_active flags to frame-level arrays.

    Each clip "owns" the frame range [start, start + clip_step). The score
    is held constant (zero-order interpolation) across those frames — the
    same approach used by RTFM and C2FPL for UCF-Crime evaluation.

    Parameters
    ----------
    clip_starts  : start frame index of each clip
    clip_scores  : normalized float32 score per clip — shape (N,)
    clip_active  : bool is_active flag per clip — shape (N,)
    total_frames : total frame count for this video
    clip_step    : raw frame span per clip (default CLIP_STEP=16)

    Returns
    -------
    (frame_scores, frame_active) — both shape (total_frames,)
    """
    frame_scores = np.zeros(total_frames, dtype=np.float32)
    frame_active = np.zeros(total_frames, dtype=bool)

    for i, start in enumerate(clip_starts):
        if start >= total_frames:
            break
        end = min(start + clip_step, total_frames)
        frame_scores[start:end] = clip_scores[i]
        frame_active[start:end] = clip_active[i]

    return frame_scores, frame_active


def build_gt_frame_array(
    annotation:   VideoAnnotation,
    total_frames: int,
) -> np.ndarray:
    """
    Build a binary (0/1) frame-level GT array of shape (total_frames,).

    Frames within any annotated temporal segment receive label 1.
    Normal videos (no segments) return an all-zeros array.
    Segment boundaries are inclusive on both ends.
    """
    gt = np.zeros(total_frames, dtype=np.int32)
    for s, e in annotation.segments:
        s_clipped = max(0, min(s, total_frames))
        e_clipped = max(0, min(e + 1, total_frames))   # inclusive → exclusive
        gt[s_clipped:e_clipped] = 1
    return gt


# ──────────────────────────────────────────────────────────────────────────────
# 4. Assemble flat arrays across all videos
# ──────────────────────────────────────────────────────────────────────────────

def assemble_frame_arrays(
    results:     List[VideoResult],
    annotations: Dict[str, VideoAnnotation],
    sat_scale:   float = SAT_SCALE,
    clip_step:   int   = CLIP_STEP,
) -> FrameArrays:
    """
    Build concatenated frame-level arrays by iterating over all scored videos.

    Score normalization (ReLU + global scale) is applied here.
    Videos missing from annotations are skipped silently.
    Videos with no clip data (empty results) are skipped for frame-level
    arrays but still counted in video-level metrics via video_accuracy().

    Returns
    -------
    FrameArrays with flat (N,) arrays across all included videos.
    """
    all_scores:    List[np.ndarray] = []
    all_gt:        List[np.ndarray] = []
    all_active:    List[np.ndarray] = []
    all_video_ids: List[np.ndarray] = []

    for vid_idx, result in enumerate(results):
        ann = annotations.get(result.video_name)
        if ann is None or not result.clips:
            continue

        starts     = [c["start"]     for c in result.clips]
        raw_scores = np.array([c["score"]     for c in result.clips], dtype=np.float32)
        is_active  = np.array([c["is_active"] for c in result.clips], dtype=bool)

        # Infer video length from last clip end (conservative proxy)
        total_frames = max(starts) + clip_step

        norm_scores = apply_relu_scale(raw_scores, sat_scale)
        frame_scores, frame_active = clips_to_frames(
            starts, norm_scores, is_active, total_frames, clip_step
        )
        gt = build_gt_frame_array(ann, total_frames)

        all_scores.append(frame_scores)
        all_gt.append(gt)
        all_active.append(frame_active)
        all_video_ids.append(np.full(total_frames, vid_idx, dtype=np.int32))

    if not all_scores:
        empty_f = np.array([], dtype=np.float32)
        empty_i = np.array([], dtype=np.int32)
        empty_b = np.array([], dtype=bool)
        return FrameArrays(empty_f, empty_i, empty_b, empty_i)

    return FrameArrays(
        scores    = np.concatenate(all_scores),
        gt_binary = np.concatenate(all_gt),
        is_active = np.concatenate(all_active),
        video_ids = np.concatenate(all_video_ids),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Frame-level metrics
# ──────────────────────────────────────────────────────────────────────────────

def frame_auc(
    scores:    np.ndarray,
    gt_binary: np.ndarray,
) -> float:
    """
    Area Under the ROC Curve at frame level.

    Returns nan if only one class is present in gt_binary.
    """
    if len(np.unique(gt_binary)) < 2:
        return float("nan")
    return float(roc_auc_score(gt_binary, scores))


def frame_ap(
    scores:    np.ndarray,
    gt_binary: np.ndarray,
) -> float:
    """
    Average Precision (area under Precision-Recall curve) at frame level.

    AP is more informative than AUC under class imbalance, which is typical
    in anomaly detection (anomalous frames are a small fraction of total).

    Returns nan if only one class is present in gt_binary.
    """
    if len(np.unique(gt_binary)) < 2:
        return float("nan")
    return float(average_precision_score(gt_binary, scores))


def roc_curve_arrays(
    scores:    np.ndarray,
    gt_binary: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (fpr, tpr, thresholds) suitable for plotting the ROC curve."""
    return roc_curve(gt_binary, scores)


def pr_curve_arrays(
    scores:    np.ndarray,
    gt_binary: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (precision, recall, thresholds) for plotting the PR curve."""
    return precision_recall_curve(gt_binary, scores)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Video-level metrics (binary ON/OFF)
# ──────────────────────────────────────────────────────────────────────────────

def video_accuracy(
    results:     List[VideoResult],
    annotations: Dict[str, VideoAnnotation],
) -> Tuple[float, Dict[str, Dict]]:
    """
    Video-level accuracy based on raw expert ON/OFF activation.

    Classification rule (mirrors production logic):
        Predicted Anomaly  ← any clip has is_active=True
        Predicted Normal   ← all clips have is_active=False (or no clips)

    Parameters
    ----------
    results     : list of VideoResult (including normal videos with empty clips)
    annotations : output of parse_gt_annotations()

    Returns
    -------
    (acc, per_video_details)
      acc               : float in [0, 1]
      per_video_details : {video_name: {gt, pred, correct, expert}}
    """
    details: Dict[str, Dict] = {}
    correct = 0
    total   = 0

    for result in results:
        ann = annotations.get(result.video_name)
        if ann is None:
            continue
        gt_anom   = ann.is_anomaly
        pred_anom = result.is_active_video
        is_correct = (gt_anom == pred_anom)
        details[result.video_name] = {
            "gt":      gt_anom,
            "pred":    pred_anom,
            "correct": is_correct,
            "expert":  result.expert_type,
        }
        correct += int(is_correct)
        total   += 1

    acc = correct / total if total > 0 else 0.0
    return float(acc), details


def class_maa(
    results:     List[VideoResult],
    annotations: Dict[str, VideoAnnotation],
) -> Tuple[float, Dict[str, float]]:
    """
    Mean Average Accuracy (mAA) — accuracy averaged per expert class.

    Normalizing per class prevents the over-represented Normal class from
    dominating the aggregate score (UCF-Crime has 950 normal videos vs
    ~100 per anomaly class in the standard split).

    Returns
    -------
    (maa, per_class_acc)
      maa           : float — mean of per-class accuracies
      per_class_acc : {expert_name: acc_float}
    """
    class_correct: Dict[str, int] = {}
    class_total:   Dict[str, int] = {}

    for result in results:
        ann = annotations.get(result.video_name)
        if ann is None:
            continue
        key = ann.expert
        class_correct.setdefault(key, 0)
        class_total.setdefault(key, 0)
        class_correct[key] += int(ann.is_anomaly == result.is_active_video)
        class_total[key]   += 1

    per_class: Dict[str, float] = {
        cls: class_correct[cls] / class_total[cls]
        for cls in class_total
        if class_total[cls] > 0
    }
    maa_val = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return maa_val, per_class


# ──────────────────────────────────────────────────────────────────────────────
# 7. Abnormal-only temporal localization analysis
# ──────────────────────────────────────────────────────────────────────────────

def abnormal_only_metrics(
    results:     List[VideoResult],
    annotations: Dict[str, VideoAnnotation],
    sat_scale:   float = SAT_SCALE,
    clip_step:   int   = CLIP_STEP,
) -> Tuple[float, float]:
    """
    AUC and AP computed exclusively within anomalous videos.

    Purpose: measures temporal localization quality — within a video that
    IS anomalous, can the system identify WHEN (which frames) the anomaly
    occurs? This is independent of whether the video was correctly classified
    at the video level.

    Returns
    -------
    (abn_auc, abn_ap) — floats in [0, 1], or nan if insufficient data
    """
    abn_scores: List[np.ndarray] = []
    abn_gt:     List[np.ndarray] = []

    for result in results:
        ann = annotations.get(result.video_name)
        if ann is None or not ann.is_anomaly or not result.clips:
            continue

        starts     = [c["start"]     for c in result.clips]
        raw_scores = np.array([c["score"]     for c in result.clips], dtype=np.float32)
        is_active  = np.array([c["is_active"] for c in result.clips], dtype=bool)

        total_frames = max(starts) + clip_step
        norm_scores  = apply_relu_scale(raw_scores, sat_scale)
        frame_scores, _ = clips_to_frames(
            starts, norm_scores, is_active, total_frames, clip_step
        )
        gt = build_gt_frame_array(ann, total_frames)

        abn_scores.append(frame_scores)
        abn_gt.append(gt)

    if not abn_scores:
        return float("nan"), float("nan")

    flat_scores = np.concatenate(abn_scores)
    flat_gt     = np.concatenate(abn_gt)

    if flat_gt.sum() == 0 or flat_gt.sum() == len(flat_gt):
        return float("nan"), float("nan")

    return frame_auc(flat_scores, flat_gt), frame_ap(flat_scores, flat_gt)


# ──────────────────────────────────────────────────────────────────────────────
# 8. Hardware benchmarking
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_hardware(
    model:       Any,
    dummy_input: Any,
    n_warmup:    int = 5,
    n_runs:      int = 50,
    device_str:  str = "cuda",
) -> Tuple[float, float, float]:
    """
    Measure FPS, GFLOPs, and peak VRAM for a given PyTorch model.

    Model-agnostic: accepts any nn.Module and a representative input tensor
    or tuple of tensors. Can be used with CLIP, VideoMAE, I3D, or any other
    backbone without modification.

    Parameters
    ----------
    model       : nn.Module — already moved to device and in eval mode
    dummy_input : torch.Tensor | tuple — representative batch input
    n_warmup    : GPU warm-up runs (not measured)
    n_runs      : timed inference runs
    device_str  : "cuda" or "cpu"

    Returns
    -------
    (fps, gflops, vram_gb)
      fps     : forward passes per second
      gflops  : GFLOPs per forward pass (via thop; 0.0 if thop unavailable)
      vram_gb : peak GPU memory allocated in GB (0.0 on CPU)
    """
    import torch

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    inp    = dummy_input if isinstance(dummy_input, tuple) else (dummy_input,)

    # ── GFLOPs via thop ──────────────────────────────────────────────────────
    gflops = 0.0
    try:
        from thop import profile as thop_profile
        macs, _ = thop_profile(model, inputs=inp, verbose=False)
        gflops  = float(macs) / 1e9
    except Exception as exc:
        print(f"[benchmark] thop unavailable or error: {exc}")

    # ── Reset VRAM stats ──────────────────────────────────────────────────────
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.eval()

    # Warm-up (JIT, CUDA lazy init, etc.)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(*inp)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Timed runs
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(*inp)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    fps     = n_runs / elapsed if elapsed > 0 else 0.0
    vram_gb = 0.0
    if device.type == "cuda":
        vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    return fps, gflops, vram_gb


# ──────────────────────────────────────────────────────────────────────────────
# 9. Convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    results:     List[VideoResult],
    annotations: Dict[str, VideoAnnotation],
    sat_scale:   float = SAT_SCALE,
    clip_step:   int   = CLIP_STEP,
) -> MetricsReport:
    """
    Compute the full VAD metric suite in one call.

    Frame-level AUC/AP are computed over scored (anomalous) videos only,
    since normal videos are not run through the expert pipeline. This is
    equivalent to the abnormal-video AUC commonly reported in zero-shot
    VAD papers. For full cross-video AUC (including normal), pass VideoResult
    objects for normal videos with score=0 arrays.

    Video-level ACC and mAA use all videos (normal videos with no clip data
    are treated as predicted-Normal, which is correct for production).

    Parameters
    ----------
    results     : list of VideoResult — include normal videos with empty clips
    annotations : output of parse_gt_annotations()
    sat_scale   : global saturation S for score normalization
    clip_step   : frames per clip for expansion

    Returns
    -------
    MetricsReport with all fields populated (hardware fields default to 0.0)
    """
    arrays = assemble_frame_arrays(results, annotations, sat_scale, clip_step)

    f_auc = frame_auc(arrays.scores, arrays.gt_binary)
    f_ap  = frame_ap(arrays.scores,  arrays.gt_binary)

    acc_val, _     = video_accuracy(results, annotations)
    maa_val, p_cls = class_maa(results, annotations)

    abn_auc, abn_ap = abnormal_only_metrics(
        results, annotations, sat_scale, clip_step
    )

    return MetricsReport(
        frame_auc     = f_auc,
        frame_ap      = f_ap,
        acc           = acc_val,
        maa           = maa_val,
        abn_auc       = abn_auc,
        abn_ap        = abn_ap,
        per_class_acc = p_cls,
    )
