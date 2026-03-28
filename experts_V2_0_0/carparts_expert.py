"""
experts_V2_0_0/carparts_expert.py
==================================
CarPartsExpert V2.0.0 — basado en CarPartsExpert_V3.py

Cambios sobre V1:
  - ALERT_THRESHOLD: 0.48 → 0.50
"""

import queue
import threading
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
import torch
import clip

from core.base_expert import BaseExpert

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
clip_device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# CONFIG — scoring temporal
# ---------------------------------------------------------------------------
SEGMENT_LEN    = 24       # frames por segmento
KEY_FRAMES_K   = 4        # 1 representativo + 3 adicionales
GAMMA1         = 0.55     # peso score_kf
GAMMA2         = 0.30     # peso score_pc
GAMMA3         = 0.15     # peso score_tc
ALERT_THRESHOLD = 0.50    # umbral sobre ascore suavizado (V2: 0.48 → 0.50)
SMOOTH_SIGMA   = 2.0
SMOOTH_HISTORY = 12

KF_SIGMOID_CENTER = 2.8
KF_SIGMOID_SCALE  = 1.5

TC_SIM_LOW  = 0.15
TC_SIM_HIGH = 0.35

# CONFIG — WinCLIP
IMG_SIZE  = 240
WIN_SMALL = (48,  48)
WIN_MID   = (80,  80)
WIN_LARGE = (120, 120)

# CONFIG — crop estabilizado
REF_W                 = 1280
REF_H                 = 720
CP_PERIMETRO_PX_REF   = 50
CP_PAD_SOLO_REF       = 15
CP_PAD_GRUPO_REF      = 40
CP_GRUPO_DIST_PX_REF  = 200
CROP_FREEZE_FRAMES    = 60
CROP_CONFIRM_FRAMES   = 5
CROP_GENEROUS_PAD_REF = 80
CROP_MOVE_TH_PX_REF   = 45
CROP_MAX_W_FRAC       = 0.65
CROP_MAX_H_FRAC       = 0.80

# CONFIG — vehículo ancla
ANCHOR_CONFIRM_FRAMES = 3
ANCHOR_MISS_MAX       = 25


# ---------------------------------------------------------------------------
# Helpers — escala de umbrales
# ---------------------------------------------------------------------------
def _scale_thresholds(frame_w: int, frame_h: int) -> dict:
    _ = frame_h
    s = frame_w / REF_W
    return {
        "perimetro":    max(10, int(CP_PERIMETRO_PX_REF  * s)),
        "pad_solo":     max(4,  int(CP_PAD_SOLO_REF      * s)),
        "pad_grupo":    max(8,  int(CP_PAD_GRUPO_REF     * s)),
        "grupo_dist":   max(40, int(CP_GRUPO_DIST_PX_REF * s)),
        "generous_pad": max(10, int(CROP_GENEROUS_PAD_REF * s)),
        "move_th":      max(8,  int(CROP_MOVE_TH_PX_REF  * s)),
        "scale":        s,
    }


# ---------------------------------------------------------------------------
# Helpers — geometría
# ---------------------------------------------------------------------------
def _dist_persona_vehiculo(box_p, box_v) -> float:
    cx = (box_p[0] + box_p[2]) / 2.0
    cy = box_p[3]
    dx = max(box_v[0] - cx, 0.0, cx - box_v[2])
    dy = max(box_v[1] - cy, 0.0, cy - box_v[3])
    return float(np.sqrt(dx * dx + dy * dy))


def _score_candidato(box_p, box_v, prev_box_p) -> float:
    cx = (box_p[0] + box_p[2]) / 2.0
    cy = box_p[3]
    dx = max(box_v[0] - cx, 0.0, cx - box_v[2])
    dy = max(box_v[1] - cy, 0.0, cy - box_v[3])
    dist      = float(np.sqrt(dx * dx + dy * dy)) + 1e-3
    prox_score = 100.0 / dist
    mov_score  = 0.0
    if prev_box_p is not None:
        xA = max(box_p[0], prev_box_p[0]); yA = max(box_p[1], prev_box_p[1])
        xB = min(box_p[2], prev_box_p[2]); yB = min(box_p[3], prev_box_p[3])
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        a1    = (box_p[2]-box_p[0]) * (box_p[3]-box_p[1])
        a2    = (prev_box_p[2]-prev_box_p[0]) * (prev_box_p[3]-prev_box_p[1])
        iou   = inter / (a1 + a2 - inter + 1e-6)
        mov_score = (1.0 - iou) * 50.0
    return prox_score + mov_score


def _compute_crop_bbox(frame, box_p, box_v, pad):
    H, W   = frame.shape[:2]
    cx_p   = (box_p[0] + box_p[2]) / 2.0
    vx_near = box_v[0] if cx_p < (box_v[0] + box_v[2]) / 2 else box_v[2]
    vy_near = box_v[1]
    x1 = int(max(0, min(box_p[0], vx_near) - pad))
    y1 = int(max(0, min(box_p[1], vy_near) - pad))
    x2 = int(min(W, max(box_p[2], vx_near) + pad))
    y2 = int(min(H, max(box_p[3], vy_near) + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2-x1) > W * CROP_MAX_W_FRAC or (y2-y1) > H * CROP_MAX_H_FRAC:
        return None
    return (x1, y1, x2, y2)


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1+x2) / 2.0, (y1+y2) / 2.0)


def _crop_from_bbox(frame, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    return crop if (crop is not None and crop.size > 0) else None


# ---------------------------------------------------------------------------
# KSM — Key frames Selection Module
# ---------------------------------------------------------------------------
def _ksm_select(
    frames_bgr: List[np.ndarray],
    text_feat:  torch.Tensor,
    clip_model,
    clip_preprocess,
) -> Tuple[int, List[int]]:
    N    = len(frames_bgr)
    imgs = []
    for fr in frames_bgr:
        small = cv2.resize(fr, (224, 224))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        imgs.append(clip_preprocess(Image.fromarray(rgb)))
    imgs_t = torch.stack(imgs).to(clip_device)

    with torch.no_grad():
        feats = clip_model.encode_image(imgs_t).float()
        feats /= feats.norm(dim=-1, keepdim=True)
        tf   = text_feat / text_feat.norm(dim=-1, keepdim=True)
        sims = (tf @ feats.T).cpu().squeeze(0).numpy()

    hat_k_idx = int(np.argmax(sims))
    offset    = hat_k_idx % max(1, N // KEY_FRAMES_K)
    other_idx = []
    for i in range(1, KEY_FRAMES_K):
        x = (i * N // KEY_FRAMES_K) + offset
        other_idx.append(min(x, N - 1))
    return hat_k_idx, other_idx


# ---------------------------------------------------------------------------
# PC — Position Context via WinCLIP multi-scale
# ---------------------------------------------------------------------------
def _compute_pc(
    key_frame_bgr: np.ndarray,
    text_feat:     torch.Tensor,
    clip_model,
    clip_preprocess,
) -> Tuple[np.ndarray, float, np.ndarray, Optional[Tuple[float,float,float,float]]]:
    """
    Retorna (M_norm, score_pc, attn_overlay_bgr, hot_bbox_norm).
    """
    img      = cv2.resize(key_frame_bgr, (IMG_SIZE, IMG_SIZE))
    all_maps = []

    for (wh, ww) in [WIN_SMALL, WIN_MID, WIN_LARGE]:
        n_rows = IMG_SIZE // wh
        n_cols = IMG_SIZE // ww
        patches = []
        for r in range(n_rows):
            for c in range(n_cols):
                patch = img[r*wh:(r+1)*wh, c*ww:(c+1)*ww]
                patches.append(
                    clip_preprocess(
                        Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                    )
                )
        if not patches:
            continue
        prep = torch.stack(patches).to(clip_device)
        with torch.no_grad():
            feats = clip_model.encode_image(prep).float()
            feats /= feats.norm(dim=-1, keepdim=True)
            tf    = text_feat / text_feat.norm(dim=-1, keepdim=True)
            sims  = (tf @ feats.T).cpu().squeeze(0).numpy().astype(np.float32)
        sim_map   = sims.reshape(n_rows, n_cols)
        mu, sigma = sim_map.mean(), sim_map.std() + 1e-8
        sim_map   = (sim_map - mu) / sigma
        all_maps.append(
            cv2.resize(sim_map, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        )

    empty_ovl = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    if not all_maps:
        return np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32), 0.0, empty_ovl, None

    M      = np.mean(all_maps, axis=0).astype(np.float32)
    M_norm = (M - M.min()) / (M.max() - M.min() + 1e-8)

    flat     = M_norm.flatten()
    half     = len(flat) // 2
    score_pc = float(np.mean(np.sort(flat)[half:]))

    # Heatmap overlay
    heatmap      = cv2.applyColorMap((M_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    attn_overlay = cv2.addWeighted(img, 0.45, heatmap, 0.55, 0)

    # Hot bbox
    rows_h, cols_h = np.where(M_norm >= 0.65)
    if len(rows_h) == 0:
        th_val = float(np.percentile(M_norm, 90))
        rows_h, cols_h = np.where(M_norm >= th_val)
    hot_bbox_norm: Optional[Tuple[float,float,float,float]] = None
    if len(rows_h) > 0:
        hot_bbox_norm = (
            float(cols_h.min()) / IMG_SIZE,
            float(rows_h.min()) / IMG_SIZE,
            float(cols_h.max() + 1) / IMG_SIZE,
            float(rows_h.max() + 1) / IMG_SIZE,
        )

    return M_norm, score_pc, attn_overlay, hot_bbox_norm


# ---------------------------------------------------------------------------
# TC — Temporal Context via Grid Image Generation
# ---------------------------------------------------------------------------
def _compute_tc(
    key_frames_bgr: List[np.ndarray],
    text_feat:      torch.Tensor,
    clip_model,
    clip_preprocess,
) -> float:
    """Retorna score_tc normalizado [0,1]."""
    if len(key_frames_bgr) < 4:
        return 0.0

    frames    = [cv2.resize(f, (IMG_SIZE, IMG_SIZE)) for f in key_frames_bgr[:4]]
    grids_pil = []

    for (wh, ww) in [WIN_SMALL, WIN_MID, WIN_LARGE]:
        n_rows = IMG_SIZE // wh
        n_cols = IMG_SIZE // ww
        for r in range(n_rows):
            for c in range(n_cols):
                patches = [f[r*wh:(r+1)*wh, c*ww:(c+1)*ww] for f in frames]
                cell  = 112
                row0  = np.hstack([cv2.resize(p, (cell, cell)) for p in patches[:2]])
                row1  = np.hstack([cv2.resize(p, (cell, cell)) for p in patches[2:]])
                grid_rgb = cv2.cvtColor(np.vstack([row0, row1]), cv2.COLOR_BGR2RGB)
                grids_pil.append(Image.fromarray(grid_rgb))

    if not grids_pil:
        return 0.0

    prep = torch.stack([clip_preprocess(g) for g in grids_pil]).to(clip_device)
    with torch.no_grad():
        feats = clip_model.encode_image(prep).float()
        feats /= feats.norm(dim=-1, keepdim=True)
        tf   = text_feat / text_feat.norm(dim=-1, keepdim=True)
        sims = (tf @ feats.T).cpu().squeeze(0).numpy()

    raw_sim  = float(sims.max())
    score_tc = float(np.clip(
        (raw_sim - TC_SIM_LOW) / (TC_SIM_HIGH - TC_SIM_LOW + 1e-8),
        0.0, 1.0
    ))
    return score_tc


# ---------------------------------------------------------------------------
# CarPartsExpert — BaseExpert
# ---------------------------------------------------------------------------
class CarPartsExpert(BaseExpert):
    """
    Detecta robo de autopartes usando scoring temporal CLIP puro (sin LVLM):
      KSM + PC (WinCLIP) + TC (grid temporal) con crop estabilizado.
    """

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------
    def __init__(self):
        # CLIP (inyectado por load())
        self._model        = None
        self._preprocess   = None
        self._text_feat    : Optional[torch.Tensor] = None   # KSM + TC
        self._pos_text     : Optional[torch.Tensor] = None   # score_kf
        self._neg_text     : Optional[torch.Tensor] = None
        self._logit_scale  : float = 100.0

        # Scoring
        self._score_history: List[float] = []
        self._smoothed_score: float      = 0.0
        self._last_ascore   : float      = 0.0
        self._last_score_kf : float      = 0.0
        self._last_score_pc : float      = 0.0
        self._last_score_tc : float      = 0.0

        # Estado público de detección
        self._is_detected  : bool  = False

        # Visualización
        self._last_kf_bgr   : Optional[np.ndarray] = None
        self._last_attn_ovl : Optional[np.ndarray] = None
        self._last_hot_bbox : Optional[Tuple[float,float,float,float]] = None
        self._last_crop_raw : Optional[np.ndarray] = None   # crop sin escalar

        # Crop estabilizado
        self._th           : Optional[dict] = None   # se inicializa en primer frame
        self._prev_boxes   : dict = {}
        self._crop_mode    : str  = "solo"

        self._anchor_v_id   : Optional[int]         = None
        self._anchor_box_v  : Optional[np.ndarray]  = None
        self._anchor_miss   : int  = 0
        self._anchor_idle   : int  = 0
        self._anchor_cand_id: Optional[int] = None
        self._anchor_cand_count: int = 0

        self._frozen_bbox   : Optional[Tuple] = None
        self._frozen_center : Optional[Tuple] = None
        self._frozen_timer  : int  = 0
        self._pending_bbox  : Optional[Tuple] = None
        self._pending_center: Optional[Tuple] = None
        self._pending_count : int  = 0
        self._crop_state    : str  = "inactivo"

        # Buffer de segmento y worker
        self._seg_buffer   : List[np.ndarray] = []
        self._processing   : bool = False
        self._q            : queue.Queue = queue.Queue(maxsize=2)

    # ------------------------------------------------------------------
    # BaseExpert — propiedades
    # ------------------------------------------------------------------
    @property
    def label(self) -> str:
        return "ROBO_AP"

    @property
    def is_active(self) -> bool:
        return self._is_detected

    # ------------------------------------------------------------------
    # BaseExpert — load
    # ------------------------------------------------------------------
    def load(self, model, preprocess) -> None:
        self._model      = model
        self._preprocess = preprocess
        self._logit_scale = float(model.logit_scale.exp().item())

        query = "a person stealing car headlights or parts from a parked vehicle"
        pos_texts = [
            "a person prying out car headlights with tools",
            "hands pulling off a vehicle headlight",
            "someone dismantling the front lights of a car",
            "a thief stealing headlights from a vehicle",
        ]
        neg_texts = [
            "a person walking past a car",
            "owner opening a car door",
            "a parked car with lights intact",
            "street background",
        ]

        with torch.no_grad():
            tok = clip.tokenize([query]).to(clip_device)
            tf  = model.encode_text(tok).float()
            tf  /= tf.norm(dim=-1, keepdim=True)
            self._text_feat = tf

            pf = model.encode_text(clip.tokenize(pos_texts).to(clip_device)).float()
            nf = model.encode_text(clip.tokenize(neg_texts).to(clip_device)).float()
            self._pos_text = (pf / pf.norm(dim=-1, keepdim=True)).mean(0, keepdim=True)
            self._neg_text = (nf / nf.norm(dim=-1, keepdim=True)).mean(0, keepdim=True)

        threading.Thread(target=self._worker_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # BaseExpert — predict (llamado por _worker_loop)
    # ------------------------------------------------------------------
    def predict(self, frames: List[np.ndarray]) -> dict:
        """
        Procesa un segmento completo: KSM → score_kf → PC → TC → ascore.
        frames: lista de crops BGR 224×224.
        """
        N = len(frames)
        if N < KEY_FRAMES_K or self._model is None:
            return {"detected": False, "score": self._smoothed_score}

        # KSM
        hat_k_idx, other_idx = _ksm_select(
            frames, self._text_feat, self._model, self._preprocess
        )
        key_frames  = [frames[i] for i in [hat_k_idx] + list(other_idx)]
        hat_k       = frames[hat_k_idx]
        self._last_kf_bgr = cv2.resize(hat_k, (IMG_SIZE, IMG_SIZE))

        # score_kf (sigmoide sobre diff pos/neg)
        with torch.no_grad():
            pil  = Image.fromarray(cv2.cvtColor(hat_k, cv2.COLOR_BGR2RGB))
            img  = self._preprocess(pil).unsqueeze(0).to(clip_device)
            feat = self._model.encode_image(img).float()
            feat /= feat.norm(dim=-1, keepdim=True)
            diff = float(self._logit_scale * (feat @ self._pos_text.T - feat @ self._neg_text.T).item())
        score_kf = float(1.0 / (1.0 + np.exp(-(diff - KF_SIGMOID_CENTER) / KF_SIGMOID_SCALE)))

        # PC
        _, score_pc, attn_ovl, hot_bbox = _compute_pc(
            hat_k, self._text_feat, self._model, self._preprocess
        )
        self._last_attn_ovl = attn_ovl
        self._last_hot_bbox = hot_bbox

        # TC
        score_tc = _compute_tc(
            key_frames, self._text_feat, self._model, self._preprocess
        )

        # ascore + Gaussian smoothing
        ascore = GAMMA1 * score_kf + GAMMA2 * score_pc + GAMMA3 * score_tc
        self._score_history.append(ascore)
        if len(self._score_history) > SMOOTH_HISTORY:
            self._score_history.pop(0)
        if len(self._score_history) >= 3:
            arr = gaussian_filter1d(
                np.array(self._score_history, dtype=np.float32), sigma=SMOOTH_SIGMA
            )
            self._smoothed_score = float(arr[-1])
        else:
            self._smoothed_score = ascore

        self._last_ascore   = ascore
        self._last_score_kf = score_kf
        self._last_score_pc = score_pc
        self._last_score_tc = score_tc

        detected = self._smoothed_score > ALERT_THRESHOLD
        print(
            f"[CP-V2] ascore={ascore:.3f}  smooth={self._smoothed_score:.3f}  "
            f"kf={score_kf:.3f}  pc={score_pc:.3f}  tc={score_tc:.3f}  "
            f"{'ALERTA' if detected else 'normal'}"
        )
        return {"detected": detected, "score": self._smoothed_score,
                "ascore": ascore, "score_kf": score_kf,
                "score_pc": score_pc, "score_tc": score_tc}

    # ------------------------------------------------------------------
    # BaseExpert — get_display_data
    # ------------------------------------------------------------------
    def get_display_data(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"color": (0, 0, 180)}

        if self._frozen_bbox is not None:
            fx1, fy1, fx2, fy2 = self._frozen_bbox
            data["bboxes"] = [(fx1, fy1, fx2 - fx1, fy2 - fy1)]

        if self._last_kf_bgr is not None:
            data["last_crop"] = self._last_kf_bgr

        data["extra_text"] = (
            f"ROBO_AP  smooth={self._smoothed_score:.3f}  "
            f"kf={self._last_score_kf:.2f}  pc={self._last_score_pc:.2f}  "
            f"tc={self._last_score_tc:.2f}  [{self._crop_state}]"
        )
        return data

    # ------------------------------------------------------------------
    # BaseExpert — process_heuristics
    # ------------------------------------------------------------------
    def process_heuristics(self, frame: np.ndarray, tracker_data: Dict[str, Any]) -> None:
        H_f, W_f = frame.shape[:2]

        # Inicializar umbrales en el primer frame
        if self._th is None:
            self._th = _scale_thresholds(W_f, H_f)

        th = self._th

        person_xyxy  = tracker_data["persons_xyxy"]
        person_ids   = tracker_data["persons_ids"]
        vehicle_xyxy = tracker_data.get("vehicles_all_xyxy", tracker_data["vehicles_xyxy"])
        vehicle_ids  = tracker_data.get("vehicles_all_ids",  tracker_data["vehicles_ids"])

        # ── Vehículo ancla ──────────────────────────────────────────────
        if self._anchor_v_id is not None:
            anchor_found = False
            for j, v_id in enumerate(vehicle_ids):
                if int(v_id) == self._anchor_v_id:
                    self._anchor_box_v = vehicle_xyxy[j].copy()
                    self._anchor_miss  = 0
                    anchor_found       = True
                    break
            if not anchor_found:
                self._anchor_miss += 1
                if self._anchor_miss > ANCHOR_MISS_MAX:
                    print(f"[CP-V2] ANCLA PERDIDA  v_id={self._anchor_v_id}  "
                          f"miss>{ANCHOR_MISS_MAX}", flush=True)
                    self._anchor_v_id = self._anchor_box_v = None
                    self._anchor_miss = self._anchor_idle = 0
                    self._anchor_cand_id    = None
                    self._anchor_cand_count = 0

        eval_v_xyxy = (np.array([self._anchor_box_v]) if self._anchor_v_id is not None
                       else vehicle_xyxy)
        eval_v_ids  = (np.array([self._anchor_v_id])  if self._anchor_v_id is not None
                       else vehicle_ids)

        # ── Candidatos ──────────────────────────────────────────────────
        candidatos = []
        for i, box_p in enumerate(person_xyxy):
            p_id = int(person_ids[i]) if i < len(person_ids) else -1
            for j, box_v in enumerate(eval_v_xyxy):
                dist = _dist_persona_vehiculo(box_p, box_v)
                if dist < th["perimetro"]:
                    v_id  = int(eval_v_ids[j]) if j < len(eval_v_ids) else -1
                    score = _score_candidato(box_p, box_v, self._prev_boxes.get(p_id))
                    candidatos.append((score, dist, box_p.copy(), box_v.copy(), p_id, v_id))

        ids_visibles = set()
        for i, box_p in enumerate(person_xyxy):
            p_id = int(person_ids[i]) if i < len(person_ids) else -1
            self._prev_boxes[p_id] = box_p.copy()
            ids_visibles.add(p_id)
        self._prev_boxes = {k: v for k, v in self._prev_boxes.items() if k in ids_visibles}

        best_box_p = best_box_v = None
        best_v_id  = None
        pad        = th["pad_solo"]
        self._crop_mode = "solo"

        if candidatos:
            candidatos.sort(key=lambda x: -x[0])
            _, _, best_box_p, best_box_v, _, best_v_id = candidatos[0]

            # Confirmación de ancla
            if self._anchor_v_id is None:
                if best_v_id == self._anchor_cand_id:
                    self._anchor_cand_count += 1
                else:
                    self._anchor_cand_id    = best_v_id
                    self._anchor_cand_count = 1
                if self._anchor_cand_count >= ANCHOR_CONFIRM_FRAMES:
                    self._anchor_v_id       = self._anchor_cand_id
                    self._anchor_box_v      = best_box_v.copy()
                    self._anchor_miss       = 0
                    self._anchor_cand_id    = None
                    self._anchor_cand_count = 0
                    print(f"[CP-V2] ANCLA CONFIRMADA  v_id={self._anchor_v_id}", flush=True)

            # Crop de grupo
            if len(candidatos) >= 2:
                box_p2 = candidatos[1][2]
                cx1 = (best_box_p[0]+best_box_p[2])/2; cy1 = (best_box_p[1]+best_box_p[3])/2
                cx2 = (box_p2[0]+box_p2[2])/2;         cy2 = (box_p2[1]+box_p2[3])/2
                if np.sqrt((cx1-cx2)**2+(cy1-cy2)**2) < th["grupo_dist"]:
                    best_box_p = np.array([
                        min(best_box_p[0], box_p2[0]), min(best_box_p[1], box_p2[1]),
                        max(best_box_p[2], box_p2[2]), max(best_box_p[3], box_p2[3]),
                    ], dtype=np.float32)
                    pad = th["pad_grupo"]
                    self._crop_mode = "grupo"
        else:
            self._anchor_cand_id    = None
            self._anchor_cand_count = 0
            if self._anchor_v_id is not None:
                self._anchor_idle += 1
                if self._anchor_idle > ANCHOR_MISS_MAX:
                    self._anchor_v_id = self._anchor_box_v = None
                    self._anchor_miss = self._anchor_idle  = 0

        if best_box_p is not None:
            self._anchor_idle = 0

        # ── Crop estabilizado ────────────────────────────────────────────
        if best_box_p is not None:
            self._frozen_timer = CROP_FREEZE_FRAMES

            if self._frozen_bbox is None:
                gen_bbox = _compute_crop_bbox(frame, best_box_p, best_box_v, th["generous_pad"])
                if gen_bbox is not None:
                    self._frozen_bbox   = gen_bbox
                    self._frozen_center = _bbox_center(gen_bbox)
                    self._pending_bbox  = None
                    self._pending_count = 0
                    self._crop_state    = "congelado"
            else:
                fx1, fy1, fx2, fy2 = self._frozen_bbox
                needs_expand = False
                ex1, ey1, ex2, ey2 = fx1, fy1, fx2, fy2
                for _, _, cp_box_p, _, _, _ in candidatos:
                    pcx = (cp_box_p[0]+cp_box_p[2])/2
                    pcy = (cp_box_p[1]+cp_box_p[3])/2
                    if pcx < fx1 or pcx > fx2 or pcy < fy1 or pcy > fy2:
                        needs_expand = True
                        ep  = th["pad_solo"]
                        ex1 = min(ex1, int(max(0,   cp_box_p[0]-ep)))
                        ey1 = min(ey1, int(max(0,   cp_box_p[1]-ep)))
                        ex2 = max(ex2, int(min(W_f, cp_box_p[2]+ep)))
                        ey2 = max(ey2, int(min(H_f, cp_box_p[3]+ep)))
                if needs_expand:
                    ew, eh = ex2-ex1, ey2-ey1
                    if ew <= W_f*CROP_MAX_W_FRAC and eh <= H_f*CROP_MAX_H_FRAC:
                        self._frozen_bbox   = (ex1, ey1, ex2, ey2)
                        self._frozen_center = _bbox_center(self._frozen_bbox)
                        self._crop_state    = "expandido"
                        self._pending_bbox  = None
                        self._pending_count = 0

                raw_bbox = _compute_crop_bbox(frame, best_box_p, best_box_v, pad)
                if raw_bbox is not None and self._crop_state != "expandido":
                    raw_center    = _bbox_center(raw_bbox)
                    dist_frozen   = np.sqrt(
                        (raw_center[0]-self._frozen_center[0])**2 +
                        (raw_center[1]-self._frozen_center[1])**2
                    )
                    if dist_frozen < th["move_th"]:
                        self._pending_bbox  = None
                        self._pending_count = 0
                        self._crop_state    = "congelado"
                    else:
                        if self._pending_bbox is None:
                            self._pending_bbox   = _compute_crop_bbox(
                                frame, best_box_p, best_box_v, th["generous_pad"])
                            self._pending_center = raw_center
                            self._pending_count  = 1
                        else:
                            d_pend = np.sqrt(
                                (raw_center[0]-self._pending_center[0])**2 +
                                (raw_center[1]-self._pending_center[1])**2
                            )
                            if d_pend < th["move_th"]:
                                self._pending_count += 1
                            else:
                                self._pending_bbox   = _compute_crop_bbox(
                                    frame, best_box_p, best_box_v, th["generous_pad"])
                                self._pending_center = raw_center
                                self._pending_count  = 1
                        self._crop_state = "pendiente"
                        if self._pending_count >= CROP_CONFIRM_FRAMES and self._pending_bbox:
                            self._frozen_bbox   = self._pending_bbox
                            self._frozen_center = self._pending_center
                            self._pending_bbox  = None
                            self._pending_count = 0
                            self._crop_state    = "congelado"
                elif self._crop_state != "expandido":
                    self._pending_bbox  = None
                    self._pending_count = 0
                    self._crop_state    = "congelado"
        else:
            self._frozen_timer = max(0, self._frozen_timer - 1)
            self._pending_bbox  = None
            self._pending_count = 0
            if self._frozen_timer == 0:
                self._frozen_bbox   = None
                self._frozen_center = None
                self._crop_state    = "inactivo"
            else:
                self._crop_state = "congelado"

        # Log de cambio de estado del crop
        new_state = self._crop_state
        if not hasattr(self, '_prev_crop_state'):
            self._prev_crop_state = new_state
        if new_state != self._prev_crop_state:
            print(f"[CP-V2] CROP_STATE  {self._prev_crop_state} → {new_state}  "
                  f"ancla={self._anchor_v_id}", flush=True)
            self._prev_crop_state = new_state

        current_crop = (_crop_from_bbox(frame, self._frozen_bbox)
                        if self._frozen_bbox is not None else None)

        # ── Buffer → worker ─────────────────────────────────────────────
        if current_crop is not None:
            self._last_crop_raw = current_crop
            self._seg_buffer.append(cv2.resize(current_crop, (224, 224)))
            if len(self._seg_buffer) >= SEGMENT_LEN:
                if not self._processing:
                    try:
                        self._q.put_nowait(list(self._seg_buffer))
                        self._processing = True
                        print(f"[CP-V2] CLIP_DISPATCH  frames={len(self._seg_buffer)}  "
                              f"ancla={self._anchor_v_id}  modo={self._crop_mode}", flush=True)
                    except queue.Full:
                        print("[CP-V2] CLIP_DISPATCH SKIP  queue llena", flush=True)
                self._seg_buffer.clear()
        else:
            self._seg_buffer.clear()
            self._smoothed_score = max(0.0, self._smoothed_score - 0.02)
            self._is_detected    = self._smoothed_score > ALERT_THRESHOLD

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            frames = self._q.get()
            try:
                res = self.predict(frames)
                self._is_detected = res["detected"]
                tag = "ALERTA" if self._is_detected else "normal"
                print(f"[CP-V2] WORKER {tag}  smooth={res['score']:.3f}  "
                      f"ascore={res.get('ascore', 0):.3f}  "
                      f"kf={res.get('score_kf', 0):.2f}  "
                      f"pc={res.get('score_pc', 0):.2f}  "
                      f"tc={res.get('score_tc', 0):.2f}", flush=True)
            except Exception as e:
                print(f"[CP-V2] Error worker: {e}", flush=True)
            self._processing = False
