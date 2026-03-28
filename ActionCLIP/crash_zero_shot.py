#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ActionCLIP zero-shot crash expert

What this does:
- Loads ActionCLIP once
- Accepts exactly 32 OpenCV BGR frames
- Produces a binary crash / non-crash decision
- Uses clip-level inference only
- No object detection
- No heuristics
- No motion rules

Dependencies:
- torch
- torchvision
- opencv-python
- pillow
- numpy
- ActionCLIP repo code available in PYTHONPATH or project:
    import clip
    from modules.Visual_Prompt import visual_prompt

Example usage (single clip):
    expert = ActionCLIPCrashExpert(
        checkpoint_path="/path/to/checkpoint.pth.tar",
        arch="ViT-B/16",
        sim_header="Transf",
        num_segments=32,
        pos_threshold=0.65,
        min_margin=0.05,
    )

    expert.load()

    result = expert.predict(frames_32_bgr)
    print(result)

Example usage (full video, sequential 32-frame clips):
    results = process_video_clips(
        expert=expert,
        video_path="/path/to/video.mp4",
        num_segments=32,
    )
    for r in results:
        print(r)

Where:
    frames_32_bgr is a Python list of exactly 32 frames from OpenCV, each as BGR np.ndarray.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T

# These imports come from the ActionCLIP repository
import clip
from modules.Visual_Prompt import visual_prompt


ACTIONCLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ActionCLIPCrashExpert:
    """
    Zero-shot clip-level crash expert using ActionCLIP.

    API style:
        - initialize with config
        - call load()
        - call predict(frames)

    Input:
        frames: list of exactly 32 OpenCV BGR frames

    Output dict:
        {
            "detected": bool,
            "set_score": float,
            "hits": int,
            "pos_prob": float,
            "neg_prob": float,
            "margin": float,
            "label": "crash" or "normal"
        }
    """

    def __init__(
        self,
        checkpoint_path: str,
        arch: str = "ViT-B/16",
        sim_header: str = "Transf",
        num_segments: int = 32,
        pos_threshold: float = 0.65,
        min_margin: float = 0.05,
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
        self.device = device or ACTIONCLIP_DEVICE

        self.model = None
        self.fusion_model = None
        self.preprocess = None
        self.text_features = None

        # Positive prompts: crash semantics
        self.pos_prompts = [
            # Impact dynamics
            "a video of two vehicles colliding on a street at night",
            "a car suddenly slamming into another vehicle at high speed",
            "a rear-end collision between two cars after a speedbump",
            # Aftermath visuals
            "a car spinning out of control and hitting another vehicle",
            # Camera/scene dynamics
            "a CCTV recording of a sudden violent car crash",
            "an aerial recording of a sudden violent car crash",
            # Crash subtypes
            "a side-impact collision between two cars at an intersection",
            "a vehicle rolling over after a high-speed crash",
        ]

        self.neg_prompts = [
            # Free-flow states
            "vehicles maintaining safe following distance on a road",
            # Urban/slow states  
            "cars stopped at a red traffic light waiting to proceed",
            "vehicles moving slowly through an urban intersection",
            # Visually distinct normal states
            "a car changing lanes safely",
            "traffic moving normally past a road sign",
            # Night / low-vis normal
            "cars driving on a road at night with headlights on",
            # Parking / very slow
            "a vehicle parking slowly in an empty parking lot",
            "vehicles parked along the side of a street",
        ]

    def _build_preprocess(self):
        scale_size = self.input_size * 256 // 224
        return T.Compose([
            T.Resize(scale_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(self.input_size),
            T.ToTensor(),
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def load(self):
        """
        Load ActionCLIP backbone, temporal fusion head, and text prompt features.
        """
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.preprocess = self._build_preprocess()

        # ActionCLIP-specific clip.load signature
        self.model, clip_state_dict = clip.load(
            self.arch,
            device=self.device,
            jit=False,
            tsm=False,
            T=self.num_segments,
            dropout=0.0,
            emb_dropout=0.0,
        )

        self.fusion_model = visual_prompt(
            self.sim_header,
            clip_state_dict,
            self.num_segments
        ).to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint does not contain 'model_state_dict'")
        if "fusion_model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint does not contain 'fusion_model_state_dict'")

        model_sd = checkpoint["model_state_dict"]
        if all(k.startswith("module.") for k in model_sd):
            model_sd = {k[len("module."):]: v for k, v in model_sd.items()}
        self.model.load_state_dict(model_sd, strict=True)

        fusion_sd = checkpoint["fusion_model_state_dict"]
        # Strip "module." prefix left by DataParallel if present
        if all(k.startswith("module.") for k in fusion_sd):
            fusion_sd = {k[len("module."):]: v for k, v in fusion_sd.items()}
        self.fusion_model.load_state_dict(fusion_sd, strict=True)

        self.model.eval()
        self.fusion_model.eval()

        with torch.no_grad():
            pos_tokens = clip.tokenize(self.pos_prompts).to(self.device)
            neg_tokens = clip.tokenize(self.neg_prompts).to(self.device)

            pos_text = self.model.encode_text(pos_tokens)
            neg_text = self.model.encode_text(neg_tokens)

            pos_text = pos_text / pos_text.norm(dim=-1, keepdim=True)
            neg_text = neg_text / neg_text.norm(dim=-1, keepdim=True)

            # Create one prototype for crash and one prototype for normal
            pos_mean = pos_text.mean(dim=0, keepdim=True)
            neg_mean = neg_text.mean(dim=0, keepdim=True)

            pos_mean = pos_mean / pos_mean.norm(dim=-1, keepdim=True)
            neg_mean = neg_mean / neg_mean.norm(dim=-1, keepdim=True)

            self.text_features = torch.cat([pos_mean, neg_mean], dim=0)

        print(f"[INFO] ActionCLIP loaded on device: {self.device}")
        print(f"[INFO] Backbone: {self.arch}")
        print(f"[INFO] Temporal head: {self.sim_header}")
        print(f"[INFO] Num segments: {self.num_segments}")
        print(f"[INFO] Checkpoint: {self.checkpoint_path}")

    def _check_loaded(self):
        if self.model is None or self.fusion_model is None or self.text_features is None:
            raise RuntimeError("Model not loaded. Call load() first.")

    def _preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Convert list of BGR OpenCV frames into tensor [T, 3, H, W].
        """
        if len(frames) != self.num_segments:
            raise ValueError(
                f"Expected exactly {self.num_segments} frames, got {len(frames)}"
            )

        processed = []
        for i, frame in enumerate(frames):
            if frame is None:
                raise ValueError(f"Frame at index {i} is None")
            if not isinstance(frame, np.ndarray):
                raise TypeError(f"Frame at index {i} is not a numpy array")
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Frame at index {i} does not have shape HxWx3")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor = self.preprocess(pil_img)
            processed.append(tensor)

        clip_tensor = torch.stack(processed, dim=0)  # [T, 3, H, W]
        return clip_tensor

    @torch.no_grad()
    def predict(self, frames: List[np.ndarray]) -> Dict:
        """
        Predict crash vs normal for a single 32-frame clip.
        """
        self._check_loaded()

        clip_tensor = self._preprocess_frames(frames).to(self.device)

        # ActionCLIP inference flow:
        # 1) encode each frame
        # 2) reshape to [B, T, D]
        # 3) fuse temporally
        # 4) compare with text features
        image_features = self.model.encode_image(clip_tensor)   # [T, D]
        image_features = image_features.unsqueeze(0)            # [1, T, D]

        video_features = self.fusion_model(image_features)      # [1, D]
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * (video_features @ text_features.T)     # [1, 2]
        probs = F.softmax(logits, dim=-1)

        pos_prob = float(probs[0, 0].item())
        neg_prob = float(probs[0, 1].item())
        margin = pos_prob - neg_prob

        detected = (pos_prob >= self.pos_threshold) and (margin >= self.min_margin)

        return {
            "detected": bool(detected),
            "set_score": round(pos_prob, 4),
            "hits": int(detected),
            "pos_prob": round(pos_prob, 4),
            "neg_prob": round(neg_prob, 4),
            "margin": round(margin, 4),
            "label": "crash" if detected else "normal",
        }


def read_all_frames(video_path: str) -> Tuple[List[np.ndarray], int, int]:
    """
    Read all frames from a video file.

    Returns:
        frames: list of all BGR frames
        width:  frame width in pixels
        height: frame height in pixels
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from video: {video_path}")

    return frames, width, height


def make_black_frame(width: int, height: int) -> np.ndarray:
    """Return a completely black BGR frame of the given dimensions."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def split_into_clips(
    frames: List[np.ndarray],
    num_segments: int,
    width: int,
    height: int,
) -> List[List[np.ndarray]]:
    """
    Split a flat list of frames into sequential non-overlapping clips of
    exactly `num_segments` frames each.

    If the last clip has fewer than `num_segments` frames it is padded with
    completely black frames (not a repeat of the last real frame).

    Returns:
        A list of clips, where each clip is a list of exactly `num_segments`
        BGR np.ndarray frames.
    """
    clips = []
    total = len(frames)
    start = 0

    while start < total:
        chunk = frames[start : start + num_segments]

        # Pad with black frames if this is a short final clip
        if len(chunk) < num_segments:
            black = make_black_frame(width, height)
            padding = [black.copy() for _ in range(num_segments - len(chunk))]
            chunk = chunk + padding

        clips.append(chunk)
        start += num_segments

    return clips


def process_video_clips(
    expert: ActionCLIPCrashExpert,
    video_path: str,
    num_segments: int = 32,
) -> List[Dict]:
    """
    Process an entire video by splitting it into sequential 32-frame clips
    and running the crash expert on each one.

    Short final clips are padded with black frames.

    Args:
        expert:       An already-loaded ActionCLIPCrashExpert instance.
        video_path:   Path to the input video file.
        num_segments: Frames per clip (default 32).

    Returns:
        A list of result dicts, one per clip, each augmented with:
            "clip_index":   int  – 0-based clip number
            "frame_start":  int  – first frame index (inclusive) in the original video
            "frame_end":    int  – last  frame index (inclusive) in the original video
            "padded_frames":int  – number of black padding frames added (0 for full clips)
    """
    print(f"[INFO] Reading all frames from: {video_path}")
    frames, width, height = read_all_frames(video_path)
    total_frames = len(frames)
    print(f"[INFO] Total frames read: {total_frames}  ({width}x{height})")

    clips = split_into_clips(frames, num_segments, width, height)
    print(f"[INFO] Total clips to process: {len(clips)}")

    results = []
    for clip_idx, clip_frames in enumerate(clips):
        frame_start = clip_idx * num_segments
        # The real (non-padded) end frame index
        frame_end = min(frame_start + num_segments - 1, total_frames - 1)
        padded = max(0, (frame_start + num_segments) - total_frames)

        print(
            f"[INFO] Processing clip {clip_idx + 1}/{len(clips)} "
            f"(frames {frame_start}–{frame_end}"
            + (f", {padded} black padding frame(s))" if padded else ")")
        )

        result = expert.predict(clip_frames)
        result["clip_index"]    = clip_idx
        result["frame_start"]   = frame_start
        result["frame_end"]     = frame_end
        result["padded_frames"] = padded

        results.append(result)

    return results


def sample_video_to_32_frames(video_path: str, num_segments: int = 32) -> List[np.ndarray]:
    """
    Helper function for testing.
    Reads a video and samples exactly 32 frames uniformly.
    Returns OpenCV BGR frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from video: {video_path}")

    if len(frames) < num_segments:
        # Pad by repeating the last frame
        while len(frames) < num_segments:
            frames.append(frames[-1].copy())
        return frames[:num_segments]

    idxs = np.linspace(0, len(frames) - 1, num_segments).astype(int)
    sampled = [frames[i] for i in idxs]
    return sampled


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ActionCLIP zero-shot crash detection")
    parser.add_argument("--checkpoint", required=True, help="Path to ActionCLIP checkpoint (.pth.tar)")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--arch", default="ViT-B/16", help="CLIP backbone architecture (default: ViT-B/16)")
    parser.add_argument("--sim-header", default="Transf", help="Temporal fusion head (default: Transf)")
    parser.add_argument("--num-segments", type=int, default=32, help="Frames per clip (default: 32)")
    parser.add_argument("--pos-threshold", type=float, default=0.65, help="Crash probability threshold (default: 0.65)")
    parser.add_argument("--min-margin", type=float, default=0.05, help="Minimum pos/neg margin (default: 0.05)")
    parser.add_argument(
        "--mode",
        choices=["single", "full"],
        default="full",
        help=(
            "Inference mode: "
            "'full' processes the entire video as sequential 32-frame clips (default); "
            "'single' samples one 32-frame clip uniformly from the whole video."
        ),
    )
    args = parser.parse_args()

    expert = ActionCLIPCrashExpert(
        checkpoint_path=args.checkpoint,
        arch=args.arch,
        sim_header=args.sim_header,
        num_segments=args.num_segments,
        pos_threshold=args.pos_threshold,
        min_margin=args.min_margin,
    )

    expert.load()

    if args.mode == "full":
        # ── Full-video mode: sequential non-overlapping 32-frame clips ──────
        results = process_video_clips(
            expert=expert,
            video_path=args.video,
            num_segments=args.num_segments,
        )

        print("\n[RESULTS]")
        crash_clips = 0
        for r in results:
            pad_note = f"  ⚠ {r['padded_frames']} black padding frame(s)" if r["padded_frames"] else ""
            print(
                f"  Clip {r['clip_index']:>4d} "
                f"[frames {r['frame_start']:>6d}–{r['frame_end']:>6d}]  "
                f"{r['label'].upper():<7}  "
                f"pos={r['pos_prob']:.4f}  neg={r['neg_prob']:.4f}  margin={r['margin']:+.4f}"
                + pad_note
            )
            if r["detected"]:
                crash_clips += 1

        print(f"\n[SUMMARY]  {crash_clips}/{len(results)} clips detected as CRASH")

    else:
        # ── Single-clip mode: uniform sample across whole video ──────────────
        frames_32 = sample_video_to_32_frames(args.video, num_segments=args.num_segments)
        result = expert.predict(frames_32)

        print("\n[RESULT]")
        print(result)