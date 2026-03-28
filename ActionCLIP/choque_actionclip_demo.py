#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crash_actionclip_demo.py
========================
End-to-end offline crash detector.

Pipeline
--------
1. Read video frame by frame into a rolling buffer (demo.py style)
2. YOLO + BotSort vehicle tracking  (VehicleTracker from tracker.py)
3. EventDetector pixel-space scoring → CandidateEvent  (event_detector.py)
4. On each CandidateEvent:
   a. Compute a tight spatial crop around the two interacting vehicles
   b. Assemble a 32-frame clip  (16 frames before + 15 after the event frame)
   c. Pass the spatially-cropped 32-frame clip to ActionCLIP for verification
5. Display dashboard (ArconteExperto_UnifiedV6 style, CHOQUE only):
      ┌─────────────────────┬─────────────────────┐
      │  Full scene (YOLO)  │  Candidate crop      │
      │  NORMAL / CHOQUE    │  (zoomed pair)        │
      └─────────────────────┴─────────────────────┘
      │         INFO BAR (score, label, frame …)   │

Usage
-----
    python crash_actionclip_demo.py \\
        --video  path/to/video.mp4 \\
        --checkpoint path/to/actionclip.pt \\
        [--model yolo11x.pt] \\
        [--device cuda:0] \\
        [--arch ViT-B/16] \\
        [--pos-threshold 0.60] \\
        [--min-margin 0.05] \\
        [--show] \\
        [--save output.mp4] \\
        [--no-loop]

Dependencies
------------
    torch torchvision opencv-python pillow numpy loguru ultralytics
    ActionCLIP repo in PYTHONPATH (provides `clip` and `modules.Visual_Prompt`)
    tracker.py, event_detector.py, config.py, frame_grabber.py  (project modules)
"""

import argparse
import time
import os
import cv2
import numpy as np
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
from loguru import logger
import logging

# Silence DEBUG-level logs from event_detector (very verbose incident tracking)
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import sys
sys.path.insert(0, "/home/jonathan/ArconteDetection/car_crash")
sys.path.insert(0, "/home/jonathan/ArconteDetection/SemanticVersion/ArconteKG_Robusto/ActionCLIP")

# ActionCLIP repo imports
import clip
from modules.Visual_Prompt import visual_prompt

# Project imports
from tracker import VehicleTracker
from event_detector import EventDetector, build_vehicle_geom, CandidateEvent
from config import PipelineConfig
from frame_grabber import Frame

# =============================================================================
# CONSTANTS
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIEW_SIZE = (640, 480)          # each panel in the 2-panel dashboard

# ActionCLIP crash prompts (tuned for spatially-cropped vehicle pairs)
CRASH_POS_PROMPTS = [
    # Impact dynamics
    "a video of two vehicles violently colliding on a road",
    "a car suddenly slamming into another vehicle at high speed",
    "a rear-end collision between two cars on a highway",
    "a side-impact collision between two cars at an intersection",
    "a car spinning out of control and hitting another vehicle",
    # Aftermath — static post-impact
    "two vehicles stopped at abnormal angles in the road after a crash",
    "cars halted in the middle of a road with visible collision damage",
    "a damaged vehicle blocking a lane after a traffic accident",
    "people standing around two crashed vehicles stopped on a road",
    # Camera types present in CCTV footage
    "a CCTV recording of a sudden violent car crash",
    "an aerial recording of a sudden violent car crash",
    # Close-up pair view (matches our spatial crop)
    "a close-up of two cars making violent contact",
    "two vehicles touching with force at close range",
]

CRASH_NEG_PROMPTS = [
    # Normal traffic states
    "cars driving smoothly along a highway at steady speed",
    "vehicles maintaining safe following distance on an open road",
    "a car changing lanes safely on a multi-lane highway",
    "traffic moving normally past a road sign",
    # Normal stops — kept to avoid false positives with parked cars
    "cars stopped at a red traffic light waiting to proceed",
    "vehicles moving slowly through an urban intersection",
    "a vehicle parking slowly in an empty parking lot",
    "vehicles parked along the side of a street",
    # Night / slow scenes
    "cars driving on a road at night with headlights on",
    "two cars stopped side by side at a traffic light",
    # Close pair that is NOT a crash (matches our spatial crop in normal cases)
    "two cars parked next to each other in a parking lot",
    "two vehicles stopped close together waiting in traffic",
]

# Detection thresholds
POS_THRESHOLD = 0.60     # softmax pos score threshold
MIN_MARGIN    = 0.05     # pos - neg margin threshold

# Palette for track IDs (from demo.py)
_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def colour_for(track_id: int):
    return _PALETTE[int(track_id) % len(_PALETTE)]


# =============================================================================
# ACTIONCLIP CRASH EXPERT  — full ActionCLIP inference as designed
# backbone encode_image → temporal fusion head → cosine sim vs text prototypes
# =============================================================================

class ActionCLIPCrashExpert:
    """
    Zero-shot crash verifier using ActionCLIP as designed:

        frames [T, 3, H, W]
            → CLIP visual encoder  → [T, D]
            → temporal fusion head → [1, D]   (Transf / temporal attention)
            → cosine sim vs [pos_proto, neg_proto]
            → softmax → pos_prob, neg_prob

    Input : list of exactly num_segments BGR np.ndarray frames
            (spatially cropped around the interacting vehicle pair)
    Output: {"detected", "pos_prob", "neg_prob", "margin", "label"}
    """

    def __init__(
        self,
        checkpoint_path: str,
        arch: str = "ViT-B/16",
        sim_header: str = "Transf",
        num_segments: int = 32,
        pos_threshold: float = POS_THRESHOLD,
        min_margin: float = MIN_MARGIN,
        input_size: int = 224,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.arch = arch
        self.sim_header = sim_header
        self.num_segments = num_segments
        self.pos_threshold = pos_threshold
        self.min_margin = min_margin
        self.input_size = input_size
        self.device = device or DEVICE

        self.model        = None
        self.fusion_model = None
        self.preprocess   = None
        self.text_features = None  # [2, D]

    def _build_preprocess(self):
        scale = self.input_size * 256 // 224
        return T.Compose([
            T.Resize(scale, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(self.input_size),
            T.ToTensor(),
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def load(self):
        """Load backbone, temporal fusion head, and encode text prototypes."""
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.preprocess = self._build_preprocess()

        # ── Backbone ──────────────────────────────────────────────────────────
        self.model, clip_state_dict = clip.load(
            self.arch,
            device=self.device,
            jit=False,
            tsm=False,
            T=self.num_segments,
            dropout=0.0,
            emb_dropout=0.0,
        )

        # ── Temporal fusion head ──────────────────────────────────────────────
        self.fusion_model = visual_prompt(
            self.sim_header,
            clip_state_dict,
            self.num_segments,
        ).to(self.device)

        # ── Load weights ──────────────────────────────────────────────────────
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Backbone (no module. prefix in this checkpoint)
        model_sd = checkpoint["model_state_dict"]
        if all(k.startswith("module.") for k in model_sd):
            model_sd = {k[len("module."):]: v for k, v in model_sd.items()}
        self.model.load_state_dict(model_sd, strict=True)

        # Fusion head (has module. prefix — DataParallel checkpoint)
        fusion_sd = checkpoint["fusion_model_state_dict"]
        if all(k.startswith("module.") for k in fusion_sd):
            fusion_sd = {k[len("module."):]: v for k, v in fusion_sd.items()}
        self.fusion_model.load_state_dict(fusion_sd, strict=True)

        self.model.eval()
        self.fusion_model.eval()

        # ── Text prototypes (encoded once) ────────────────────────────────────
        with torch.no_grad():
            pos_tok = clip.tokenize(CRASH_POS_PROMPTS).to(self.device)
            neg_tok = clip.tokenize(CRASH_NEG_PROMPTS).to(self.device)

            pos_feat = self.model.encode_text(pos_tok)
            neg_feat = self.model.encode_text(neg_tok)

            pos_feat = pos_feat / pos_feat.norm(dim=-1, keepdim=True)
            neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)

            pos_proto = pos_feat.mean(dim=0, keepdim=True)
            neg_proto = neg_feat.mean(dim=0, keepdim=True)

            pos_proto = pos_proto / pos_proto.norm(dim=-1, keepdim=True)
            neg_proto = neg_proto / neg_proto.norm(dim=-1, keepdim=True)

            self.text_features = torch.cat([pos_proto, neg_proto], dim=0)  # [2, D]

        logger.info(f"[ActionCLIP] Backbone: {self.arch}  Fusion: {self.sim_header}  "
                    f"T={self.num_segments}  device={self.device}")
        logger.info(f"[ActionCLIP] Checkpoint: {self.checkpoint_path}")

    def _preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Convert list of BGR frames → tensor [T, 3, H, W]."""
        processed = []
        black = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        for fr in frames:
            if fr is None or fr.size == 0:
                fr = black
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            processed.append(self.preprocess(Image.fromarray(rgb)))
        return torch.stack(processed, dim=0)  # [T, 3, H, W]

    @torch.no_grad()
    def predict(self, frames: List[np.ndarray]) -> Dict:
        """
        Full ActionCLIP inference on a single spatiotemporally cropped clip.

        Flow:
            [T, 3, H, W] → encode_image → [T, D]
                         → unsqueeze(0) → [1, T, D]
                         → fusion_model → [1, D]
                         → cosine sim   → logits [1, 2]
                         → softmax      → pos_prob, neg_prob
        """
        if len(frames) != self.num_segments:
            # Pad or trim to exactly num_segments
            black = np.zeros((VIEW_SIZE[1], VIEW_SIZE[0], 3), dtype=np.uint8)
            if len(frames) < self.num_segments:
                frames = frames + [black.copy()] * (self.num_segments - len(frames))
            else:
                frames = frames[:self.num_segments]

        clip_tensor = self._preprocess_frames(frames).to(self.device)  # [T, 3, H, W]

        # 1) Per-frame visual features
        image_features = self.model.encode_image(clip_tensor)          # [T, D]
        image_features = image_features.unsqueeze(0)                   # [1, T, D]

        # 2) Temporal fusion
        video_features = self.fusion_model(image_features)             # [1, D]
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        # 3) Text features (already normalised)
        text_features = self.text_features                             # [2, D]

        # 4) Cosine similarity → softmax
        logits = 100.0 * (video_features @ text_features.T)           # [1, 2]
        probs  = F.softmax(logits, dim=-1)

        pos_prob = float(probs[0, 0].item())
        neg_prob = float(probs[0, 1].item())
        margin   = pos_prob - neg_prob
        detected = (pos_prob >= self.pos_threshold) and (margin >= self.min_margin)

        return {
            "detected":  bool(detected),
            "pos_prob":  round(pos_prob, 4),
            "neg_prob":  round(neg_prob, 4),
            "margin":    round(margin, 4),
            "label":     "crash" if detected else "normal",
        }



# =============================================================================
# SPATIOTEMPORAL CROP BUILDER
# =============================================================================

def compute_pair_bbox(
    b1: np.ndarray,
    b2: np.ndarray,
    pad_px: int,
    frame_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """
    Return (x1, y1, x2, y2) bounding box around both vehicles with padding,
    clamped to frame boundaries.
    """
    H, W = frame_shape
    x1 = max(0, int(min(b1[0], b2[0])) - pad_px)
    y1 = max(0, int(min(b1[1], b2[1])) - pad_px)
    x2 = min(W, int(max(b1[2], b2[2])) + pad_px)
    y2 = min(H, int(max(b1[3], b2[3])) + pad_px)
    return x1, y1, x2, y2


def crop_frame(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop and resize a frame to VIEW_SIZE."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return np.zeros((VIEW_SIZE[1], VIEW_SIZE[0], 3), dtype=np.uint8)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((VIEW_SIZE[1], VIEW_SIZE[0], 3), dtype=np.uint8)
    return cv2.resize(crop, VIEW_SIZE)


def build_clip_from_buffer(
    frame_buffer: deque,          # deque of Frame objects (global_index, image)
    event_global_idx: int,        # global frame index of the CandidateEvent
    pair_bbox: Tuple[int, int, int, int],
    num_segments: int = 32,
    pre_frames: int = 16,
) -> List[np.ndarray]:
    """
    Extract a 32-frame spatially-cropped clip centred on the event.

    - Takes `pre_frames` frames before the event frame and
      (num_segments - pre_frames - 1) frames after.
    - Frames missing from the buffer are padded with black frames.
    - Each frame is cropped to `pair_bbox` and resized to VIEW_SIZE.

    Returns list of num_segments BGR np.ndarray frames (VIEW_SIZE).
    """
    post_frames = num_segments - pre_frames - 1  # 15 frames after event

    # Build a lookup: global_index → image
    buf_dict = {f.global_index: f.image for f in frame_buffer}

    black = np.zeros((VIEW_SIZE[1], VIEW_SIZE[0], 3), dtype=np.uint8)
    result = []

    for offset in range(-pre_frames, post_frames + 1):
        idx = event_global_idx + offset
        if idx in buf_dict:
            result.append(crop_frame(buf_dict[idx], pair_bbox))
        else:
            result.append(black.copy())

    return result


# =============================================================================
# DRAWING HELPERS
# =============================================================================

def draw_tracked_vehicles(frame, tracked_vehicles, track_histories=None):
    vis = frame.copy()
    for tv in tracked_vehicles:
        if tv.mask is None:
            continue
        col = colour_for(tv.track_id)
        overlay = vis.copy()
        overlay[tv.mask == 1] = [int(c * 0.6) for c in col]
        cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)
        contours, _ = cv2.findContours(tv.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, col, 1)

    for tv in tracked_vehicles:
        x1, y1, x2, y2 = tv.bbox_xyxy.astype(int)
        col = colour_for(tv.track_id)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        label = f"#{tv.track_id} {tv.class_name} {tv.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if track_histories and tv.track_id in track_histories:
            pts = [p.center.astype(int) for p in track_histories[tv.track_id].points[-30:]]
            for k in range(1, len(pts)):
                alpha = k / len(pts)
                c = tuple(int(v * alpha) for v in col)
                cv2.line(vis, tuple(pts[k-1]), tuple(pts[k]), c, 2)
    return vis


def draw_event_line(frame, event: CandidateEvent, geoms_dict: Dict):
    """Draw a line + circles between the two vehicle centres."""
    vis = frame.copy()
    id1, id2 = event.pair_key
    g1, g2 = geoms_dict.get(id1), geoms_dict.get(id2)
    if g1 is not None and g2 is not None:
        pt1 = (int(g1.center[0]), int(g1.center[1]))
        pt2 = (int(g2.center[0]), int(g2.center[1]))
        cv2.line(vis, pt1, pt2, (0, 0, 255), 3)
        cv2.circle(vis, pt1, 7, (0, 0, 255), -1)
        cv2.circle(vis, pt2, 7, (0, 0, 255), -1)
    return vis


def draw_scene_panel(
    frame: np.ndarray,
    tracked_vehicles,
    track_histories,
    crash_confirmed: bool,
    clip_result: Optional[Dict],
    active_event: Optional[CandidateEvent],
    geoms_dict: Dict,
    frame_idx: int,
) -> np.ndarray:
    """
    Left panel: full scene with YOLO overlays and NORMAL / CHOQUE status bar.
    Style from ArconteExperto_UnifiedV6._draw_status_on_rgb.
    """
    vis = draw_tracked_vehicles(frame, tracked_vehicles, track_histories)
    vis = cv2.resize(vis, VIEW_SIZE)

    # Draw event line if candidate active
    if active_event is not None and geoms_dict:
        # Scale geom centres to VIEW_SIZE
        h_orig, w_orig = frame.shape[:2]
        sx = VIEW_SIZE[0] / w_orig
        sy = VIEW_SIZE[1] / h_orig
        id1, id2 = active_event.pair_key
        g1, g2 = geoms_dict.get(id1), geoms_dict.get(id2)
        if g1 is not None and g2 is not None:
            pt1 = (int(g1.center[0] * sx), int(g1.center[1] * sy))
            pt2 = (int(g2.center[0] * sx), int(g2.center[1] * sy))
            cv2.line(vis, pt1, pt2, (0, 0, 255), 3)
            cv2.circle(vis, pt1, 7, (0, 0, 255), -1)
            cv2.circle(vis, pt2, 7, (0, 0, 255), -1)

    # Status bar (top)
    if crash_confirmed:
        cv2.rectangle(vis, (0, 0), (VIEW_SIZE[0], 44), (0, 0, 120), -1)
        cv2.putText(vis, "CHOQUE", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(vis, (0, 0), (VIEW_SIZE[0] - 1, VIEW_SIZE[1] - 1), (0, 0, 255), 4)
    elif active_event is not None:
        cv2.rectangle(vis, (0, 0), (VIEW_SIZE[0], 44), (0, 60, 120), -1)
        cv2.putText(vis, f"Candidato score={active_event.score:.2f}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(vis, (0, 0), (VIEW_SIZE[0], 44), (0, 80, 0), -1)
        cv2.putText(vis, "NORMAL", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    # Frame counter (bottom-right)
    cv2.putText(vis, f"F:{frame_idx}", (VIEW_SIZE[0] - 90, VIEW_SIZE[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return vis


def draw_crop_panel(
    crop: Optional[np.ndarray],
    crash_confirmed: bool,
    clip_result: Optional[Dict],
    event: Optional[CandidateEvent],
    ttl: int,
    clip_frame_idx: Optional[int] = None,
    clip_total: Optional[int] = None,
) -> np.ndarray:
    """
    Right panel: animated 32-frame ActionCLIP clip replay with result overlay.
    Shows each spatially-cropped frame in sequence, cycling continuously.
    """
    if crop is not None:
        vis = cv2.resize(crop.copy(), VIEW_SIZE)
    else:
        vis = np.zeros((VIEW_SIZE[1], VIEW_SIZE[0], 3), dtype=np.uint8)
        cv2.putText(vis, "Sin candidato", (160, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2, cv2.LINE_AA)

    # Label bar
    if crash_confirmed and clip_result is not None:
        bar_color = (0, 0, 120)
        label = (f"CHOQUE  pos={clip_result['pos_prob']:.4f}  "
                 f"margin={clip_result['margin']:+.4f}  ttl:{ttl}")
        txt_color = (0, 0, 255)
        cv2.rectangle(vis, (0, 0), (VIEW_SIZE[0] - 1, VIEW_SIZE[1] - 1), (0, 0, 255), 4)
    elif clip_result is not None and not crash_confirmed:
        bar_color = (0, 50, 80)
        label = (f"NORMAL  pos={clip_result['pos_prob']:.4f}  "
                 f"margin={clip_result['margin']:+.4f}")
        txt_color = (0, 165, 255)
    elif event is not None:
        bar_color = (30, 30, 0)
        rel = {"end_end": "EE", "end_side": "ES", "side_side": "SS"}.get(
            event.relation, "??")
        label = f"Candidato #{event.pair_key[0]}-#{event.pair_key[1]}  {rel}  score={event.score:.2f}"
        txt_color = (0, 200, 255)
    else:
        bar_color = (0, 40, 0)
        label = "Sin crash"
        txt_color = (0, 200, 0)

    cv2.rectangle(vis, (0, 0), (VIEW_SIZE[0], 36), bar_color, -1)
    cv2.putText(vis, label, (6, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, txt_color, 1, cv2.LINE_AA)

    # Clip replay progress bar + frame counter (bottom)
    if clip_frame_idx is not None and clip_total and clip_total > 0:
        disp_idx = clip_frame_idx % clip_total
        progress = int(VIEW_SIZE[0] * disp_idx / clip_total)
        bar_y = VIEW_SIZE[1] - 14
        cv2.rectangle(vis, (0, bar_y), (VIEW_SIZE[0], VIEW_SIZE[1]), (20, 20, 20), -1)
        cv2.rectangle(vis, (0, bar_y), (progress, VIEW_SIZE[1]), (0, 180, 255), -1)
        counter = f"Clip [{disp_idx+1}/{clip_total}]"
        cv2.putText(vis, counter, (VIEW_SIZE[0] - 110, VIEW_SIZE[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_info_bar(
    crash_confirmed: bool,
    clip_result: Optional[Dict],
    active_event: Optional[CandidateEvent],
    fps: float,
    frame_idx: int,
    clips_verified: int,
    ttl: int,
) -> np.ndarray:
    """
    Bottom info bar (1280 wide × 80 tall), style from ArconteExperto_UnifiedV6.
    """
    bar = np.zeros((80, VIEW_SIZE[0], 3), dtype=np.uint8)
    bar_color = (0, 0, 180) if crash_confirmed else (0, 100, 0)
    cv2.rectangle(bar, (0, 0), (bar.shape[1], 80), bar_color, -1)

    status = "CHOQUE" if crash_confirmed else "NORMAL"

    if clip_result:
        r = clip_result
        line1 = (f"{status}  |  pos={r['pos_prob']:.4f}  neg={r['neg_prob']:.4f}  "
                 f"margin={r['margin']:+.4f}  label={r['label'].upper()}")
    else:
        line1 = f"{status}  |  Sin resultado CLIP"

    if active_event:
        id1, id2 = active_event.pair_key
        rel = active_event.relation
        line2 = (f"FPS:{fps:.1f}  F:{frame_idx}  CLIPs:{clips_verified}  ttl:{ttl}  "
                 f"Candidato #{id1}-#{id2}  rel={rel}  event_score={active_event.score:.2f}")
    else:
        line2 = f"FPS:{fps:.1f}  F:{frame_idx}  CLIPs verificados:{clips_verified}"

    cv2.putText(bar, line1, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, line2, (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return bar


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Crash detection: EventDetector → ActionCLIP")
    p.add_argument("--video",       required=True,       help="Input video path")
    p.add_argument("--checkpoint",  required=True,       help="ActionCLIP checkpoint (.pt / .pth.tar)")
    p.add_argument("--model",       default="yolo11x.pt",help="YOLO model path")
    p.add_argument("--device",      default="cuda:0",    help="Torch device")
    p.add_argument("--arch",        default="ViT-B/16",  help="CLIP backbone arch")
    p.add_argument("--clip-size",   type=int, default=30,help="Frames per detection clip")
    p.add_argument("--overlap",     type=int, default=15,help="Overlap between detection clips")
    p.add_argument("--conf",        type=float, default=0.4, help="YOLO confidence threshold")
    p.add_argument("--pad",         type=int, default=80, help="Pixel padding around vehicle pair crop")
    p.add_argument("--pre-frames",  type=int, default=16,
                   help="Frames before event for 32-frame ActionCLIP clip")
    p.add_argument("--pos-threshold", type=float, default=POS_THRESHOLD)
    p.add_argument("--min-margin",    type=float, default=MIN_MARGIN)
    p.add_argument("--min-score",   type=float, default=0.0,
                   help="Minimum EventDetector score to pass candidate to ActionCLIP")
    p.add_argument("--crash-ttl",   type=int, default=90,
                   help="Frames to hold CHOQUE label after confirmation")
    p.add_argument("--show",        action="store_true", help="Display live window")
    p.add_argument("--save",        default=None,        help="Save annotated video to path")
    p.add_argument("--no-loop",     action="store_true", help="Do not loop video")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load ActionCLIP ─────────────────────────────────────────────────────
    logger.info("Loading ActionCLIP crash expert…")
    expert = ActionCLIPCrashExpert(
        checkpoint_path=args.checkpoint,
        arch=args.arch,
        num_segments=32,
        pos_threshold=args.pos_threshold,
        min_margin=args.min_margin,
        device=args.device,
    )
    expert.load()

    # ── Load YOLO tracker ───────────────────────────────────────────────────
    logger.info(f"Loading YOLO model: {args.model} on {args.device}")
    tracker = VehicleTracker(
        model_path=args.model,
        tracker_config="botsort.yaml",
        device=args.device,
    )

    # ── Event detector ──────────────────────────────────────────────────────
    cfg = PipelineConfig()
    event_detector = EventDetector(cfg)

    # ── Video source ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.video}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video: {args.video}  {vid_w}x{vid_h} @ {source_fps:.1f}fps")

    # ── Video writer ────────────────────────────────────────────────────────
    writer = None
    if args.save:
        dash_w = VIEW_SIZE[0]               # 640
        dash_h = VIEW_SIZE[1] + 80         # 560
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, source_fps, (dash_w, dash_h))
        logger.info(f"Saving output to: {args.save}")

    # ── Rolling frame buffer (large enough for pre+post clip assembly) ──────
    PRE_FRAMES = args.pre_frames
    POST_FRAMES = 32 - PRE_FRAMES - 1     # 15
    buf_capacity = args.clip_size + (PRE_FRAMES + POST_FRAMES + 10) * 2
    frame_buffer: deque = deque(maxlen=buf_capacity)

    # ── Detection clip parameters ────────────────────────────────────────────
    clip_size = args.clip_size
    overlap   = args.overlap
    stride    = clip_size - overlap

    global_idx   = 0
    clip_id      = 0
    last_end_idx = -1

    # ── Pending events: events we are waiting to gather post-frames for ──────
    # Each entry: {"event": CandidateEvent, "pair_bbox": tuple, "ready_at": int}
    pending_events: List[Dict] = []

    # ── Display state ────────────────────────────────────────────────────────
    crash_confirmed = False
    crash_ttl       = 0
    last_clip_result: Optional[Dict] = None
    last_crop: Optional[np.ndarray]  = None
    last_clip_frames: Optional[List[np.ndarray]] = None   # full 32-frame clip for replay
    clip_replay_idx: int = 0                              # current frame in replay
    active_event: Optional[CandidateEvent] = None
    clips_verified = 0

    t_last = time.monotonic()
    fps_smooth = 0.0

    # ── Deduplicate events (same pair within 45 frames) ──────────────────────
    last_event_frame: Dict[Tuple, int] = {}
    EVENT_COOLDOWN = 45

    logger.info("Starting detection loop. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            if args.no_loop:
                logger.info("Video ended.")
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            last_end_idx = -1
            tracker.reset()
            continue

        f = Frame(
            image=frame,
            timestamp=global_idx / source_fps,
            global_index=global_idx,
            source_fps=source_fps,
        )
        frame_buffer.append(f)
        global_idx += 1

        # ── Check if any pending event is ready (post_frames collected) ──────
        now_idx = global_idx - 1
        still_pending = []
        for pe in pending_events:
            if now_idx >= pe["ready_at"]:
                # Build 32-frame spatially cropped clip
                clip_frames = build_clip_from_buffer(
                    frame_buffer=frame_buffer,
                    event_global_idx=pe["event"].frame_idx,
                    pair_bbox=pe["pair_bbox"],
                    num_segments=32,
                    pre_frames=PRE_FRAMES,
                )

                # Run ActionCLIP
                result = expert.predict(clip_frames)
                clips_verified += 1
                last_clip_result = result
                last_crop = clip_frames[PRE_FRAMES]   # show the event frame crop
                last_clip_frames = clip_frames        # store full 32-frame clip for replay
                clip_replay_idx = 0                   # restart replay from frame 0
                active_event = pe["event"]

                rel = pe["event"].relation
                id1, id2 = pe["event"].pair_key
                logger.info(
                    f"[ActionCLIP] Clip #{clips_verified}  pair={id1}-{id2}  "
                    f"rel={rel}  label={result['label'].upper()}  "
                    f"pos={result['pos_prob']:.4f}  neg={result['neg_prob']:.4f}  "
                    f"margin={result['margin']:+.4f}"
                )

                if result["detected"]:
                    crash_confirmed = True
                    crash_ttl = args.crash_ttl
                    logger.warning(
                        f"  *** CHOQUE CONFIRMADO ***  pair={id1}-{id2}  "
                        f"event_frame={pe['event'].frame_idx}  "
                        f"pos={result['pos_prob']:.4f}  margin={result['margin']:+.4f}"
                    )
            else:
                still_pending.append(pe)
        pending_events = still_pending

        # ── Only run detection every `stride` frames once buffer is full ─────
        if len(frame_buffer) < clip_size:
            continue
        latest_idx = frame_buffer[-1].global_index
        if last_end_idx >= 0 and (latest_idx - last_end_idx) < stride:
            # Still draw dashboard at every frame when we have state
            pass
        else:
            # ── Build detection clip ──────────────────────────────────────────
            frames_list  = list(frame_buffer)[-clip_size:]
            clip_frames_ = [f.image for f in frames_list]
            clip_times   = [f.timestamp for f in frames_list]
            last_end_idx = frames_list[-1].global_index

            # Track
            tracked_per_frame = tracker.process_clip(clip_frames_, clip_times, persist=True)

            # Detect events
            events = event_detector.process_clip(tracked_per_frame, clip_times)

            # Build geoms for last frame (for line drawing)
            geoms_last: Dict = {}
            for tv in tracked_per_frame[-1]:
                try:
                    g = build_vehicle_geom(tv, clip_frames_[-1].shape[:2], cfg)
                    geoms_last[tv.track_id] = g
                except Exception:
                    pass

            # ── Process each emitted event ────────────────────────────────────
            for ev in events:
                pair_key = ev.pair_key
                last_fi  = last_event_frame.get(pair_key, -9999)
                if (ev.frame_idx - last_fi) < EVENT_COOLDOWN:
                    continue        # cooldown — skip duplicate
                last_event_frame[pair_key] = ev.frame_idx

                # Determine spatial bbox for the two vehicles
                id1, id2 = pair_key
                g1 = geoms_last.get(id1)
                g2 = geoms_last.get(id2)

                if g1 is not None and g2 is not None:
                    # Use geom centres ± half-extents as rough bboxes
                    def geom_to_box(g):
                        cx, cy = g.center
                        hw = abs(g.major_extent[1] - g.major_extent[0]) / 2.0
                        hh = abs(g.minor_extent[1] - g.minor_extent[0]) / 2.0
                        size = max(hw, hh, 30)
                        return np.array([cx - size, cy - size, cx + size, cy + size])

                    b1 = geom_to_box(g1)
                    b2 = geom_to_box(g2)
                else:
                    # Fallback: find vehicle bboxes from last-frame tracking
                    boxes = {tv.track_id: tv.bbox_xyxy
                             for tv in tracked_per_frame[-1]}
                    b1 = boxes.get(id1)
                    b2 = boxes.get(id2)
                    if b1 is None or b2 is None:
                        # Try any frame in the clip
                        for frame_tvs in reversed(tracked_per_frame):
                            bmap = {tv.track_id: tv.bbox_xyxy for tv in frame_tvs}
                            if id1 in bmap and id2 in bmap:
                                b1, b2 = bmap[id1], bmap[id2]
                                break
                        if b1 is None or b2 is None:
                            logger.warning(f"[Event] Cannot find bboxes for pair {pair_key}, skipping")
                            continue

                pair_bbox = compute_pair_bbox(
                    np.array(b1[:4]), np.array(b2[:4]),
                    pad_px=args.pad,
                    frame_shape=(vid_h, vid_w),
                )

                # Show the crop immediately (pre-ActionCLIP)
                event_raw_frame = clip_frames_[min(ev.frame_idx, len(clip_frames_) - 1)]
                last_crop = crop_frame(event_raw_frame, pair_bbox)
                active_event = ev

                # Filter by minimum EventDetector score
                if ev.score < args.min_score:
                    logger.info(
                        f"[Candidate] SKIPPED pair={pair_key} score={ev.score:.2f} "
                        f"< min_score={args.min_score:.2f}"
                    )
                    continue

                # Schedule ActionCLIP inference once post_frames are collected
                ready_at = ev.frame_idx + POST_FRAMES + 1
                pending_events.append({
                    "event":     ev,
                    "pair_bbox": pair_bbox,
                    "ready_at":  ready_at,
                })

                logger.info(
                    f"[Candidate] frame={ev.frame_idx}  pair={pair_key}  "
                    f"rel={ev.relation}  score={ev.score:.2f}  "
                    f"bbox={pair_bbox}  → ActionCLIP at frame {ready_at}"
                )

            clip_id += 1

        # ── TTL countdown ────────────────────────────────────────────────────
        if crash_confirmed:
            crash_ttl -= 1
            if crash_ttl <= 0:
                crash_confirmed = False
                crash_ttl       = 0

        # ── FPS ──────────────────────────────────────────────────────────────
        now = time.monotonic()
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(now - t_last, 1e-6))
        t_last = now

        # ── Build dashboard ─────────────────────────────────────────────────
        # LEFT panel: always the true live frame (real-time stream)
        # RIGHT panel: replay the full 32-frame ActionCLIP clip one frame per
        #              dashboard tick, cycling until a new clip arrives.
        vis_tracked = tracked_per_frame[-1] if "tracked_per_frame" in dir() else []
        vis_geoms   = geoms_last            if "geoms_last"         in dir() else {}

        # Left — clean live stream, no annotations
        scene_vis = draw_scene_panel(
            frame=frame,
            tracked_vehicles=[],
            track_histories=None,
            crash_confirmed=crash_confirmed,
            clip_result=last_clip_result,
            active_event=None,
            geoms_dict={},
            frame_idx=global_idx,
        )



        info = draw_info_bar(
            crash_confirmed=crash_confirmed,
            clip_result=last_clip_result,
            active_event=active_event,
            fps=fps_smooth,
            frame_idx=global_idx,
            clips_verified=clips_verified,
            ttl=crash_ttl,
        )

        dashboard = np.vstack((scene_vis, info))

        if writer:
            writer.write(dashboard)

        if args.show:
            cv2.imshow("Crash ActionCLIP", dashboard)
            delay = max(1, int(1000 / source_fps))
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), 27):
                break

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    logger.info(f"Done. Total frames: {global_idx}  "
                f"ActionCLIP verifications: {clips_verified}")


if __name__ == "__main__":
    main()