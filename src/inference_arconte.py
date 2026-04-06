#!/usr/bin/env python3
"""
src/inference_arconte.py
========================
Mass evaluation script for Arconte V2.0.0 — Zero-Shot VAD framework.

Processes videos DIRECTLY using all 5 Arconte experts (Fight, Crash, Fire,
Robbery, CarParts). Does NOT depend on a pre-computed expert_scores.npy cache.

Pipeline per video
------------------
  1. Read all raw frames
  2. For each clip (CLIP_STEP=16 frames, sampled at STRIDE=2):
       · tracker.process(frame)        — YOLO + ByteTrack, per frame
       · expert.process_heuristics()   — spatial filtering, per frame
       · expert.predict(clip_frames)   — CLIP scoring on the 16 resized frames
  3. Aggregate: score = max(score_i) over all 5 experts per clip
                is_active = any expert.is_active after the frame loop
  4. Apply ReLU + global scale S=5.0  →  score ∈ [0, 1]
  5. Generate per-video diagnostic plot immediately
  6. After all videos: compute AUC, AP, ACC, mAA via metrics_library

Usage
-----
    python src/inference_arconte.py
    python src/inference_arconte.py --video-root /path/to/UCF-Crime/Videos
    python src/inference_arconte.py --limit 20 --no-plots
    python src/inference_arconte.py --no-heuristics  # pure CLIP scoring, no tracker
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import clip as clip_lib

from core.tracker import ArconteTracker
from experts_V2_0_0.carparts_expert import CarPartsExpert
from experts_V2_0_0.crash_expert import CrashExpert
from experts_V2_0_0.fight_expert import FightExpert
from experts_V2_0_0.fire_expert import make_fire_smoke_experts
from experts_V2_0_0.robbery_expert import RobberyExpert

from metrics_library import (
    CLIP_STEP,
    CLIP_LEN,
    STRIDE,
    SAT_SCALE,
    MetricsReport,
    VideoAnnotation,
    VideoResult,
    apply_relu_scale,
    clips_to_frames,
    compute_all_metrics,
    parse_gt_annotations,
)

# ── Default paths ─────────────────────────────────────────────────────────────
_ANNOT_DEFAULT      = _ROOT / "resources" / "Temporal_Anomaly_Annotation.txt"
_OUTPUT_DEFAULT     = _ROOT / "data" / "eval_results"
_VIDEO_ROOT_DEFAULT = Path("/media/pc/backup1/BaseDeDatos/UCF-Crime/Videos")
_CLIP_CKPT_DEFAULT  = _ROOT / ".checkpoints" / "ViT-L-14.pt"
_ROBBERY_CKPT       = _ROOT / ".checkpoints" / "ViT-B-16-32-f.pt"


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _quiet():
    """Suppress stdout (silences YOLO / CLIP verbose output)."""
    with open(os.devnull, "w") as devnull:
        old, sys.stdout = sys.stdout, devnull
        try:
            yield
        finally:
            sys.stdout = old


def _resize224(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (224, 224))


# ──────────────────────────────────────────────────────────────────────────────
# Expert state reset
# ──────────────────────────────────────────────────────────────────────────────

# Known internal state variables to zero-out between videos.
# Using hasattr so this is safe even if an expert doesn't have a given var.
_RESET_SCALARS: List[Tuple[str, Any]] = [
    # CLIP-based smoothed scoring (FightExpert, CrashExpert, FireExpert, CarPartsExpert)
    ("_smoothed_score",     0.0),
    ("_consec_pos",         0.0),
    ("_last_score",         0.0),
    ("_is_detected",        False),
    # FightExpert target-lock
    ("_target_locked",      False),
    ("_locked_ids",         None),
    ("_sticky_counter",     0),
    ("_last_ids",           (-1, -1)),
    ("_last_crop",          None),
    ("_last_valid_bbox",    None),
    # CrashExpert
    ("_ttl",                0),
    ("_crash_frames",       0),
    ("_consec_run",         0),
    ("_stationary_frames",  0),
    # FireExpert
    ("_fire_ttl",           0),
    ("_fire_consec",        0),
    # CarPartsExpert anchor state
    ("_anchor_confirmed",   False),
    ("_anchor_id",          None),
    ("_anchor_miss",        0),
    ("_alert_active",       False),
    ("_alert_score",        0.0),
    ("_crop_frozen",        None),
    ("_crop_confirm_cnt",   0),
]

_RESET_COLLECTIONS: List[str] = [
    # Buffers and histories
    "_buffer", "_score_history", "_motion_history",
    "_closing_history", "_prev_boxes",
]


def _reset_expert_state(expert: Any) -> None:
    """
    Reset internal stateful variables of an expert between videos.

    Zeroes EMA smoothing, consecutive counters, crop buffers, and lock state.
    The CLIP model and text embeddings are left intact (expensive to reload).
    The worker daemon thread keeps running — it will block on queue.get()
    until the next clip is dispatched.
    """
    for name, default in _RESET_SCALARS:
        if hasattr(expert, name):
            setattr(expert, name, default)

    for name in _RESET_COLLECTIONS:
        if not hasattr(expert, name):
            continue
        obj = getattr(expert, name)
        if isinstance(obj, dict):
            setattr(expert, name, {})
        elif isinstance(obj, list):
            setattr(expert, name, [])
        elif isinstance(obj, deque):
            obj.clear()
        elif obj is not None:
            setattr(expert, name, None)


# ──────────────────────────────────────────────────────────────────────────────
# ArconteEvaluator
# ──────────────────────────────────────────────────────────────────────────────

class ArconteEvaluator:
    """
    Loads all 5 Arconte experts once and evaluates one video at a time.

    Scoring mirrors the production pipeline (process_heuristics + predict)
    but runs ALL 5 experts on EVERY video instead of just the matched one.

    Per-clip score  = max(expert.predict(frames)["score"] for all experts)
    Per-clip active = any(expert.is_active for all experts after heuristics loop)
    """

    def __init__(
        self,
        clip_ckpt:    Path = _CLIP_CKPT_DEFAULT,
        robbery_ckpt: Path = _ROBBERY_CKPT,
        device:       str  = "cuda",
    ) -> None:
        self._device = device if torch.cuda.is_available() else "cpu"
        logging.info("Loading CLIP model from %s on %s...", clip_ckpt, self._device)

        with _quiet():
            self._clip_model, self._clip_preprocess = clip_lib.load(
                str(clip_ckpt), device=self._device, jit=False
            )
        self._clip_model.eval()

        fire_expert, _ = make_fire_smoke_experts()
        self._experts: Dict[str, Any] = {
            "fight":    FightExpert(),
            "crash":    CrashExpert(),
            "fire":     fire_expert,
            "robbery":  RobberyExpert(checkpoint_path=str(robbery_ckpt)),
            "carparts": CarPartsExpert(),
        }

        logging.info("Loading expert text embeddings...")
        with _quiet():
            for name, exp in self._experts.items():
                exp.load(self._clip_model, self._clip_preprocess)
                logging.info("  [%s] loaded", name)

        logging.info("All 5 experts ready.")

    # ──────────────────────────────────────────────────────────────────────────

    def _reset_all(self) -> None:
        """Reset every expert's stateful variables before processing a new video."""
        for exp in self._experts.values():
            _reset_expert_state(exp)

    # ──────────────────────────────────────────────────────────────────────────

    def score_video(
        self,
        video_path:      Path,
        use_heuristics:  bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Score one video clip-by-clip using all 5 experts.

        Parameters
        ----------
        video_path      : path to the video file
        use_heuristics  : if True, run tracker + process_heuristics() per frame
                          to get is_active flags (mirrors production).
                          If False, derive is_active from predict()["detected"]
                          only (faster, no tracker needed).

        Returns
        -------
        List of dicts with keys:
            clip_name : str   "{stem}_s{start:06d}"
            start     : int   first raw-frame index of the clip
            is_active : bool  True if any expert fired this clip
            score     : float raw max score across all experts (un-normalized)
        """
        self._reset_all()

        if not video_path.exists():
            logging.warning("Video not found: %s", video_path)
            return []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.warning("Cannot open video: %s", video_path)
            return []

        raw_frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame)
        cap.release()

        min_frames = CLIP_LEN * STRIDE
        if len(raw_frames) < min_frames:
            logging.warning("Video too short (%d frames): %s",
                            len(raw_frames), video_path.name)
            return []

        stem    = video_path.stem
        results: List[Dict[str, Any]] = []

        # Fresh tracker per video (resets ByteTrack IDs)
        tracker = ArconteTracker() if use_heuristics else None

        for start in range(0, len(raw_frames) - min_frames + 1, CLIP_STEP):
            src_idx = [start + i * STRIDE for i in range(CLIP_LEN)]

            # ── Phase 1: heuristics (spatial filtering, fills internal buffers)
            clip_active = False
            last_td: Dict[str, Any] = {}

            if use_heuristics and tracker is not None:
                for idx in src_idx:
                    frame = raw_frames[idx]
                    try:
                        with _quiet():
                            td = tracker.process(frame)
                        last_td = td
                    except Exception:
                        td = last_td

                    for exp in self._experts.values():
                        try:
                            with _quiet():
                                exp.process_heuristics(frame, td)
                        except Exception:
                            pass
                        if exp.is_active:
                            clip_active = True

            # ── Phase 2: CLIP scoring on the full (resized) clip frames
            resized = [_resize224(raw_frames[i]) for i in src_idx]
            max_score = 0.0

            for exp in self._experts.values():
                try:
                    with _quiet():
                        res = exp.predict(resized)
                    score   = float(res.get("score",    0.0))
                    max_score = max(max_score, score)
                    # Fallback is_active via predict() if heuristics disabled
                    if not use_heuristics and res.get("detected", False):
                        clip_active = True
                except Exception as exc:
                    logging.debug("predict() error in %s: %s",
                                  type(exp).__name__, exc)

            results.append({
                "clip_name": f"{stem}_s{start:06d}",
                "start":     start,
                "is_active": clip_active,
                "score":     max_score,
            })

        n_det = sum(1 for r in results if r["is_active"])
        logging.info(
            "  %s — %d clips | %d active (%.0f%%)",
            stem, len(results), n_det,
            100.0 * n_det / max(len(results), 1),
        )
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Video file discovery
# ──────────────────────────────────────────────────────────────────────────────

def find_video_path(
    video_name: str,
    ann:        VideoAnnotation,
    video_root: Path,
) -> Optional[Path]:
    """
    Locate a UCF-Crime video file by trying common sub-directory structures.

    Tries:
      <video_root>/<video_name>.{mp4,avi}
      <video_root>/<Category>/<video_name>.{mp4,avi}
    """
    for suffix in ("mp4", "avi"):
        for subdir in ("", ann.category):
            p = video_root / subdir / f"{video_name}.{suffix}"
            if p.exists():
                return p
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Per-video diagnostic plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_video(
    result:       VideoResult,
    annotation:   VideoAnnotation,
    out_path:     Path,
    sat_scale:    float = SAT_SCALE,
    clip_step:    int   = CLIP_STEP,
    total_frames: Optional[int] = None,
) -> None:
    """
    Generate and save a diagnostic plot for one video showing:

      1. Anomaly score  — continuous blue line in [0, 1]
      2. Ground-truth segments  — green shaded area
      3. Alert ON/OFF state     — red step function from is_active flags

    The plot title includes GT category, system prediction, and a ✓/✗ verdict.
    """
    clips = result.clips
    if not clips:
        return

    starts     = [c["start"]     for c in clips]
    raw_scores = np.array([c["score"]     for c in clips], dtype=np.float32)
    is_active  = np.array([c["is_active"] for c in clips], dtype=bool)

    n_frames    = total_frames or (max(starts) + clip_step)
    norm_scores = apply_relu_scale(raw_scores, sat_scale)

    frame_scores, frame_active = clips_to_frames(
        starts, norm_scores, is_active, n_frames, clip_step
    )

    x = np.arange(n_frames)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Ground-truth shaded regions
    for seg_idx, (seg_s, seg_e) in enumerate(annotation.segments):
        label = "Ground Truth" if seg_idx == 0 else "_nolegend_"
        ax.axvspan(seg_s, min(seg_e, n_frames - 1),
                   alpha=0.20, color="limegreen", label=label)

    # Continuous anomaly score
    ax.plot(x, frame_scores, color="#1f77b4", linewidth=1.2,
            label="Anomaly Score", zorder=3)

    # ON/OFF alert as a step function (scaled to 0.90 for visibility)
    ax.step(x, frame_active.astype(np.float32) * 0.90,
            color="crimson", linewidth=0.9, alpha=0.80,
            where="post", label="Alert ON/OFF", zorder=4)

    # Legend
    gt_patch   = mpatches.Patch(color="limegreen", alpha=0.35, label="Ground Truth")
    score_line = plt.Line2D([0], [0], color="#1f77b4", lw=1.5, label="Anomaly Score")
    alert_line = plt.Line2D([0], [0], color="crimson", lw=1.0, alpha=0.8,
                             label="Alert ON/OFF")
    ax.legend(handles=[gt_patch, score_line, alert_line],
              loc="upper right", fontsize=9, framealpha=0.75)

    correct = (result.is_active_video == annotation.is_anomaly)
    verdict = "✓" if correct else "✗"
    pred_str = "ANOMALY" if result.is_active_video else "NORMAL"
    gt_str   = annotation.category if annotation.is_anomaly else "Normal"

    ax.set_xlim(0, n_frames)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Frame index", fontsize=10)
    ax.set_ylabel("Score [0–1]", fontsize=10)
    ax.set_title(
        f"{result.video_name}   GT: {gt_str}   Pred: {pred_str}   {verdict}",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Annotation of GT frame boundaries
    for seg_s, seg_e in annotation.segments:
        ax.axvline(seg_s, color="green", lw=0.7, linestyle="--", alpha=0.5)
        ax.axvline(seg_e, color="green", lw=0.7, linestyle="--", alpha=0.5)

    # Anomaly-score statistics in bottom-right
    if norm_scores.size > 0:
        peak = float(norm_scores.max())
        mean = float(norm_scores.mean())
        ax.text(
            0.99, 0.04,
            f"peak={peak:.3f}  mean={mean:.3f}",
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="gray",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Summary aggregate plots (ROC + PR)
# ──────────────────────────────────────────────────────────────────────────────

def _save_roc_pr_plots(
    frame_scores: np.ndarray,
    frame_gt:     np.ndarray,
    out_dir:      Path,
) -> None:
    from sklearn.metrics import auc, roc_curve, precision_recall_curve

    if len(np.unique(frame_gt)) < 2:
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(frame_gt, frame_scores)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, "#1f77b4", lw=2, label=f"AUC = {roc_auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlim=(0, 1), ylim=(0, 1.02),
           xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC — Arconte V2.0.0 (frame level)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "roc_curve.png", dpi=120)
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(frame_gt, frame_scores)
    pr_auc_val   = auc(rec, prec)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, "crimson", lw=2, label=f"AP = {pr_auc_val:.4f}")
    ax.set(xlim=(0, 1), ylim=(0, 1.02),
           xlabel="Recall", ylabel="Precision",
           title="Precision-Recall — Arconte V2.0.0 (frame level)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "pr_curve.png", dpi=120)
    plt.close(fig)

    logging.info("ROC and PR curves saved to %s/plots/", out_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def print_report(report: MetricsReport, n_videos: int) -> None:
    sep = "=" * 65
    print()
    print(sep)
    print("  ARCONTE V2.0.0 — Evaluation Report")
    print(sep)
    print(f"  Videos processed         : {n_videos}")
    print()

    def _f(v: float) -> str:
        return f"{v:.4f}" if v == v else "   n/a"   # nan guard

    print("  Frame-level Metrics")
    print(f"    AUC-ROC              : {_f(report.frame_auc)}")
    print(f"    AP  (PR-AUC)         : {_f(report.frame_ap)}")
    print()
    print("  Video-level Metrics  (ON/OFF — all videos)")
    print(f"    Accuracy  (ACC)      : {_f(report.acc)}  "
          f"({report.acc * 100:.1f}%)")
    print(f"    Mean Avg Acc  (mAA)  : {_f(report.maa)}  "
          f"({report.maa * 100:.1f}%)")
    print()
    print("  Abnormal-Only Localization")
    print(f"    AUC-ROC              : {_f(report.abn_auc)}")
    print(f"    AP  (PR-AUC)         : {_f(report.abn_ap)}")
    print()
    if report.per_class_acc:
        print("  Per-class ACC")
        for cls, acc_c in sorted(report.per_class_acc.items()):
            bar = "█" * int(acc_c * 20)
            print(f"    {cls:<14}: {_f(acc_c)}  {bar}")
        print()
    print(sep)
    print()


def _save_report_json(
    report:   MetricsReport,
    n_videos: int,
    out_path: Path,
) -> None:
    import json
    data = {
        "n_videos":      n_videos,
        "frame_auc":     report.frame_auc,
        "frame_ap":      report.frame_ap,
        "acc":           report.acc,
        "maa":           report.maa,
        "abn_auc":       report.abn_auc,
        "abn_ap":        report.abn_ap,
        "per_class_acc": report.per_class_acc,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logging.info("Report saved to %s", out_path)


def _save_confusion_csv(
    results:     List[VideoResult],
    annotations: Dict[str, VideoAnnotation],
    out_path:    Path,
) -> None:
    rows = []
    for r in results:
        ann = annotations.get(r.video_name)
        if ann is None:
            continue
        n_clips  = len(r.clips)
        n_active = sum(1 for c in r.clips if c["is_active"])
        max_raw  = max((c["score"] for c in r.clips), default=0.0)
        rows.append((
            r.video_name, ann.expert, ann.category,
            int(ann.is_anomaly), int(r.is_active_video),
            int(ann.is_anomaly == r.is_active_video),
            n_clips, n_active, f"{max_raw:.3f}",
        ))
    # Errors first
    rows.sort(key=lambda x: (x[5], x[2]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("video,expert,category,gt,pred,correct,n_clips,n_active,max_score\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    logging.info("Confusion details saved to %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Arconte V2.0.0 — direct video evaluation with all 5 experts"
    )
    p.add_argument("--annot",        type=Path, default=_ANNOT_DEFAULT,
                   help="Temporal_Anomaly_Annotation.txt path")
    p.add_argument("--video-root",   type=Path, default=_VIDEO_ROOT_DEFAULT,
                   help="Root directory of UCF-Crime video files")
    p.add_argument("--clip-ckpt",    type=Path, default=_CLIP_CKPT_DEFAULT,
                   help="CLIP checkpoint path (ViT-L-14.pt)")
    p.add_argument("--output",       type=Path, default=_OUTPUT_DEFAULT,
                   help="Output directory for plots and report")
    p.add_argument("--sat-scale",    type=float, default=SAT_SCALE,
                   help=f"Global saturation scale S (default {SAT_SCALE})")
    p.add_argument("--limit",        type=int, default=0,
                   help="Process at most N videos (0 = all)")
    p.add_argument("--no-plots",     action="store_true",
                   help="Skip per-video diagnostic plots (faster)")
    p.add_argument("--no-heuristics", action="store_true",
                   help="Skip tracker + process_heuristics; derive is_active "
                        "from predict()[\"detected\"] only (faster, no YOLO needed)")
    p.add_argument("--device",       type=str, default="cuda",
                   help="Torch device (default: cuda)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────────
    fmt  = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh   = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    fh   = logging.FileHandler(args.output / "eval.log", encoding="utf-8")
    fh.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(sh)
    root.addHandler(fh)
    root.setLevel(logging.INFO)

    # ── Parse GT annotations ──────────────────────────────────────────────────
    annotations: Dict[str, VideoAnnotation] = parse_gt_annotations(args.annot)
    logging.info("Annotations: %d videos", len(annotations))

    # ── Build video list (annotation ∩ existing files) ─────────────────────
    video_items: List[Tuple[str, Path]] = []
    for stem, ann in sorted(annotations.items()):
        vp = find_video_path(stem, ann, args.video_root)
        if vp is None:
            logging.debug("Video not found, skipped: %s", stem)
            continue
        video_items.append((stem, vp))

    if args.limit > 0:
        video_items = video_items[: args.limit]

    logging.info(
        "Videos to process: %d  (found in %s)",
        len(video_items), args.video_root,
    )
    if not video_items:
        logging.error(
            "No videos found under %s. Check --video-root.", args.video_root
        )
        sys.exit(1)

    # ── Load experts ──────────────────────────────────────────────────────────
    evaluator = ArconteEvaluator(
        clip_ckpt    = args.clip_ckpt,
        robbery_ckpt = _ROBBERY_CKPT,
        device       = args.device,
    )

    # ── Per-video evaluation loop ─────────────────────────────────────────────
    all_results: List[VideoResult] = []
    use_heuristics = not args.no_heuristics

    for i, (stem, video_path) in enumerate(video_items):
        ann = annotations[stem]
        logging.info(
            "[%d/%d] Processing %s  (%s)",
            i + 1, len(video_items), stem, ann.category,
        )
        t0 = time.perf_counter()

        # Score video
        clip_results = evaluator.score_video(video_path, use_heuristics)

        elapsed = time.perf_counter() - t0

        # Wrap in VideoResult
        result = VideoResult(
            video_name  = stem,
            expert_type = ann.expert,
            clips       = clip_results,
        )
        all_results.append(result)

        logging.info(
            "  → %d clips, %.1f s  |  video pred: %s  |  GT: %s",
            len(clip_results), elapsed,
            "ANOMALY" if result.is_active_video else "NORMAL",
            ann.category if ann.is_anomaly else "Normal",
        )

        # Generate plot immediately after each video
        if not args.no_plots and clip_results:
            plot_out = args.output / "plots" / ann.expert / f"{stem}.png"
            try:
                plot_video(
                    result, ann, plot_out,
                    sat_scale = args.sat_scale,
                    clip_step = CLIP_STEP,
                )
            except Exception as exc:
                logging.warning("Plot failed for %s: %s", stem, exc)

    # Add normal videos that have no associated video file (treat as TN)
    scored_stems = {r.video_name for r in all_results}
    for stem, ann in annotations.items():
        if ann.expert == "normal" and stem not in scored_stems:
            all_results.append(VideoResult(
                video_name  = stem,
                expert_type = "normal",
                clips       = [],
            ))

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    logging.info("Computing aggregate metrics over %d videos...", len(all_results))
    report = compute_all_metrics(all_results, annotations, args.sat_scale)

    # ── Print and persist ─────────────────────────────────────────────────────
    print_report(report, len(all_results))
    _save_report_json(report, len(all_results), args.output / "metrics.json")
    _save_confusion_csv(all_results, annotations, args.output / "confusion_details.csv")

    # Summary ROC / PR plots
    if not args.no_plots:
        from metrics_library import assemble_frame_arrays
        arrays = assemble_frame_arrays(all_results, annotations, args.sat_scale)
        if len(arrays.scores) > 0:
            _save_roc_pr_plots(arrays.scores, arrays.gt_binary, args.output)

    logging.info("DONE — all outputs in %s", args.output)


if __name__ == "__main__":
    main()
