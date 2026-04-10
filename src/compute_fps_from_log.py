#!/usr/bin/env python3
"""
src/compute_fps_from_log.py
===========================
Calcula FPS y coste computacional a partir del log ya generado por
inference_arconte.py. NO re-ejecuta el modelo.

Estrategia:
  1. Parsea eval.log  →  extrae video_name, n_clips, elapsed_s por video
  2. Abre cada video con OpenCV solo para leer CAP_PROP_FRAME_COUNT (ms por video)
  3. Calcula FPS = n_frames / elapsed_s  y  s/clip
  4. Imprime resumen global y tabla por video

Uso:
    python src/compute_fps_from_log.py
    python src/compute_fps_from_log.py --log data/eval_results/eval.log \
        --video-root /media/pc/backup1/BaseDeDatos/UCF-Crime/Videos
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent

_LOG_DEFAULT        = _ROOT / "data" / "eval_results" / "eval.log"
_VIDEO_ROOT_DEFAULT = Path("/media/pc/backup1/BaseDeDatos/UCF-Crime/Videos")

# Regex: "  → 87 clips, 111.8 s  |  video pred: ANOMALY  |  GT: Abuse"
_RE_RESULT = re.compile(
    r"\[INFO\] \[(\d+)/\d+\] Processing (\S+)\s"  # [N/M] Processing stem
)
_RE_TIMING = re.compile(
    r"\[INFO\]\s+→\s+(\d+)\s+clips,\s+([\d.]+)\s+s"  # → N clips, X.X s
)


def parse_log(log_path: Path) -> list[dict]:
    """
    Returns list of dicts with keys:
        video_name, n_clips, elapsed_s
    """
    records = []
    current_name = None

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = _RE_RESULT.search(line)
            if m:
                current_name = m.group(2)
                continue

            m = _RE_TIMING.search(line)
            if m and current_name:
                records.append({
                    "video_name": current_name,
                    "n_clips":    int(m.group(1)),
                    "elapsed_s":  float(m.group(2)),
                })
                current_name = None

    return records


def get_frame_count(video_name: str, video_root: Path) -> int | None:
    """
    Opens the video only to read total frame count (no frame decode).
    Returns None if not found.
    """
    for suffix in ("mp4", "avi"):
        # Try flat structure first, then category subdirs
        for subdir in video_root.iterdir() if video_root.exists() else []:
            if not subdir.is_dir():
                continue
            p = subdir / f"{video_name}.{suffix}"
            if p.exists():
                cap = cv2.VideoCapture(str(p))
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                return n if n > 0 else None
        # flat
        p = video_root / f"{video_name}.{suffix}"
        if p.exists():
            cap = cv2.VideoCapture(str(p))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return n if n > 0 else None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="FPS/cost from existing eval.log")
    ap.add_argument("--log",        type=Path, default=_LOG_DEFAULT)
    ap.add_argument("--video-root", type=Path, default=_VIDEO_ROOT_DEFAULT)
    ap.add_argument("--no-video",   action="store_true",
                    help="Skip video frame-count lookup; estimate from clips")
    args = ap.parse_args()

    if not args.log.exists():
        print(f"ERROR: log not found at {args.log}", file=sys.stderr)
        sys.exit(1)

    records = parse_log(args.log)
    if not records:
        print("ERROR: no timing entries found in log", file=sys.stderr)
        sys.exit(1)

    print(f"\nParsed {len(records)} videos from {args.log}\n")

    # CLIP_STEP=16, CLIP_LEN=16, STRIDE=2 → min_frames=32
    CLIP_STEP = 16
    CLIP_LEN  = 16
    STRIDE    = 2
    MIN_FRAMES = CLIP_LEN * STRIDE

    rows = []
    for rec in records:
        name      = rec["video_name"]
        n_clips   = rec["n_clips"]
        elapsed_s = rec["elapsed_s"]

        if args.no_video:
            # Estimate: start_last = (n_clips-1)*CLIP_STEP  →  n_frames ≈ start_last + MIN_FRAMES
            n_frames = (n_clips - 1) * CLIP_STEP + MIN_FRAMES
            source   = "est"
        else:
            n_frames = get_frame_count(name, args.video_root)
            if n_frames is None:
                n_frames = (n_clips - 1) * CLIP_STEP + MIN_FRAMES
                source   = "est"
            else:
                source = "cv2"

        fps       = n_frames / elapsed_s if elapsed_s > 0 else float("nan")
        s_per_clip = elapsed_s / n_clips  if n_clips  > 0 else float("nan")

        rows.append({
            "name":       name,
            "n_frames":   n_frames,
            "n_clips":    n_clips,
            "elapsed_s":  elapsed_s,
            "fps":        fps,
            "s_per_clip": s_per_clip,
            "source":     source,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    total_frames  = sum(r["n_frames"]  for r in rows)
    total_elapsed = sum(r["elapsed_s"] for r in rows)
    total_clips   = sum(r["n_clips"]   for r in rows)
    global_fps    = total_frames / total_elapsed if total_elapsed > 0 else float("nan")
    global_spc    = total_elapsed / total_clips  if total_clips   > 0 else float("nan")

    all_fps = [r["fps"] for r in rows if r["fps"] == r["fps"]]  # nan filter

    sep = "=" * 72
    print(sep)
    print("  FPS & Computational Cost — Arconte V2.0.0 (post-hoc)")
    print(sep)
    print(f"  Videos analysed          : {len(rows)}")
    print(f"  Total frames processed   : {total_frames:,}")
    print(f"  Total clips processed    : {total_clips:,}")
    print(f"  Total wall-clock time    : {total_elapsed/3600:.2f} h  "
          f"({total_elapsed:.0f} s)")
    print()
    print(f"  Global FPS (frames/s)    : {global_fps:.2f}")
    print(f"  Global s / clip          : {global_spc:.3f} s")
    print()
    print(f"  Per-video FPS  — mean    : {np.mean(all_fps):.2f}")
    print(f"                   median  : {np.median(all_fps):.2f}")
    print(f"                   min     : {np.min(all_fps):.2f}")
    print(f"                   max     : {np.max(all_fps):.2f}")
    print(f"                   std     : {np.std(all_fps):.2f}")
    print(sep)

    # ── Per-video table ───────────────────────────────────────────────────────
    print()
    print(f"{'Video':<35} {'Frames':>7} {'Clips':>6} {'Time(s)':>8} "
          f"{'FPS':>7} {'s/clip':>7} {'src':>4}")
    print("-" * 72)
    for r in rows:
        print(
            f"{r['name']:<35} {r['n_frames']:>7,} {r['n_clips']:>6} "
            f"{r['elapsed_s']:>8.1f} {r['fps']:>7.2f} "
            f"{r['s_per_clip']:>7.3f} {r['source']:>4}"
        )

    print("-" * 72)
    avg_fps   = np.mean(all_fps)
    avg_spc   = np.mean([r["s_per_clip"] for r in rows])
    print(
        f"{'PROMEDIO':<35} {int(total_frames/len(rows)):>7,} "
        f"{int(total_clips/len(rows)):>6} "
        f"{total_elapsed/len(rows):>8.1f} "
        f"{avg_fps:>7.2f} {avg_spc:>7.3f}"
    )
    print()


if __name__ == "__main__":
    main()
