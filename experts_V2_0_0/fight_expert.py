"""
experts_V2_0_0/fight_expert.py
================================
FightExpert V2.0.0 — basado en FightDetector_Debug.py

Mejoras sobre V1:
  - Scoring de pares: proximidad + IoU + movimiento inter-frame
    (pares estáticos como saludos ya no disparan CLIP)
  - Gate de movimiento: descarta pares casi estáticos antes de CLIP
  - Target-lock: una vez confirmado un par, se le da seguimiento
    exclusivo hasta que el score baje
  - PAD_CLOSE: crop más generoso cuando los cuerpos se solapan (IoU > 0.05)
  - Fast-track: un frame con diff >= F_FAST_TRACK_TH confirma inmediatamente
  - EMA suavizado del score (F_SMOOTH_FACTOR)
  - Prompts más específicos (8 pos + 8 neg)
"""

import queue
import threading
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import clip

from core.base_expert import BaseExpert

# ---------------------------------------------------------------------------
clip_device = "cuda" if torch.cuda.is_available() else "cpu"
VIEW_SIZE   = (640, 480)

# ---------------------------------------------------------------------------
# Constantes — heurística espacial
# ---------------------------------------------------------------------------
F_DIST_MAX_LIMIT   = 300    # distancia máx entre centros para considerar par (px)
F_MIN_IOU_CRITICAL = 0.01   # IoU mínimo para par solapado
F_STICKY_FRAMES    = 25     # frames que se mantiene el crop tras perder el par
F_PAD_PIXELS       = 100    # padding del crop (normal)
F_PAD_CLOSE        = 160    # padding extra cuando IoU > 0.05 (muy juntos)

# Gate de movimiento
F_MIN_MOVEMENT_FOR_CLIP = 4.0   # movimiento mínimo del par para enviar a CLIP

# ---------------------------------------------------------------------------
# Constantes — CLIP
# ---------------------------------------------------------------------------
F_BUFFER_SIZE    = 8
F_DIFF_HIT_TH    = 3.5
F_SET_SCORE_TH   = 3.8
F_MIN_POS_FRAMES = 3
F_CONSEC_MIN     = 3
F_SMOOTH_FACTOR  = 0.45
F_FAST_TRACK_TH  = 7.0
F_LOCK_SCORE_TH  = 4.0


class FightExpert(BaseExpert):
    """
    Detecta peleas callejeras. Mejoras sobre V1:
      1. Pair scoring (proximidad + IoU + movimiento) para elegir el par más sospechoso.
      2. Movement gate: pares estáticos (saludos, conversación) se ignoran.
      3. Target-lock: par confirmado se sigue exclusivamente.
      4. Fast-track: diff alto en un solo frame confirma inmediatamente.
    """

    # ------------------------------------------------------------------
    def __init__(self):
        # Estado de crop
        self._buffer:          list        = []
        self._last_crop:       Optional[np.ndarray] = None
        self._last_valid_bbox: Optional[Tuple]      = None   # (b1, b2, pad)
        self._sticky_counter:  int         = 0
        self._last_ids:        Tuple       = (-1, -1)

        # Target-lock
        self._target_locked: bool          = False
        self._locked_ids:    Optional[Tuple] = None

        # Movimiento inter-frame
        self._prev_boxes: Dict[int, np.ndarray] = {}

        # Estado de detección
        self._is_detected: bool  = False
        self._consec_pos:  float = 0.0
        self._last_score:  float = 0.0

        # CLIP
        self._model         = None
        self._preprocess    = None
        self._pos_text      = None
        self._neg_text      = None
        self._smoothed_score: float = 0.0

        # Worker
        self._q:           queue.Queue = queue.Queue(maxsize=2)
        self._processing:  bool        = False

    # ------------------------------------------------------------------
    @property
    def label(self) -> str:
        return "PELEA"

    @property
    def is_active(self) -> bool:
        return self._is_detected

    # ------------------------------------------------------------------
    def load(self, model, preprocess) -> None:
        self._model      = model
        self._preprocess = preprocess

        with torch.no_grad():
            pos = [
                "two people throwing punches and hitting each other hard",
                "violent street brawl with people kicking and punching aggressively",
                "two men fighting violently on the street throwing fists",
                "person striking another person with a punch or kick attack",
                "people wrestling aggressively on the ground fighting",
                "violent physical assault with hitting kicking and striking",
                "street fight with people grabbing and hitting each other",
                "two people in a violent altercation throwing blows at each other",
            ]
            neg = [
                "two people shaking hands and greeting each other calmly",
                "friends having a calm conversation standing close together",
                "two people hugging each other warmly and peacefully",
                "people standing still and talking face to face quietly",
                "two persons walking slowly side by side on the street",
                "pedestrians standing near each other waiting calmly",
                "people posing together for a photo smiling",
                "two friends meeting and greeting with a handshake or hug",
            ]
            pf = model.encode_text(clip.tokenize(pos).to(clip_device))
            nf = model.encode_text(clip.tokenize(neg).to(clip_device))
            self._pos_text = (pf / pf.norm(dim=-1, keepdim=True)).mean(0, keepdim=True)
            self._neg_text = (nf / nf.norm(dim=-1, keepdim=True)).mean(0, keepdim=True)

        threading.Thread(target=self._worker_loop, daemon=True).start()

    # ------------------------------------------------------------------
    def process_heuristics(self, frame: np.ndarray, tracker_data: Dict[str, Any]) -> None:
        self._process_fight(
            frame,
            tracker_data["persons_xyxy"],
            tracker_data["persons_ids"],
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, frames: List[np.ndarray]) -> dict:
        scale = float(self._model.logit_scale.exp().item())
        diffs: List[float] = []
        hits = 0

        for fr in frames:
            if fr is None or fr.size == 0:
                continue
            pil  = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            img  = self._preprocess(pil).unsqueeze(0).to(clip_device)
            feat = self._model.encode_image(img)
            feat /= feat.norm(dim=-1, keepdim=True)
            diff = float(scale * (feat @ self._pos_text.T - feat @ self._neg_text.T).item())
            diffs.append(diff)

            # Fast-track: evidencia muy clara en un solo frame
            if diff >= F_FAST_TRACK_TH:
                self._smoothed_score = diff
                return {
                    "detected":   True,
                    "score":      self._smoothed_score,
                    "hits":       1,
                    "raw":        diff,
                    "fast_track": True,
                }
            if diff > F_DIFF_HIT_TH:
                hits += 1

        if not diffs:
            return {"detected": False, "score": 0.0, "hits": 0, "raw": 0.0, "fast_track": False}

        raw_score = (float(np.sort(diffs)[-3:].mean())
                     if len(diffs) >= 3 else float(np.mean(diffs)))
        self._smoothed_score = (F_SMOOTH_FACTOR * raw_score +
                                (1 - F_SMOOTH_FACTOR) * self._smoothed_score)
        detected = (hits >= F_MIN_POS_FRAMES) and (self._smoothed_score > F_SET_SCORE_TH)
        return {
            "detected":   detected,
            "score":      self._smoothed_score,
            "hits":       hits,
            "raw":        raw_score,
            "fast_track": False,
        }

    # ------------------------------------------------------------------
    def get_display_data(self) -> Dict[str, Any]:
        return {"last_crop": self._last_crop} if self._last_crop is not None else {}

    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            frames = self._q.get()
            try:
                res = self.predict(frames)
                self._last_score = res["score"]

                if res.get("fast_track"):
                    self._consec_pos = 10.0
                    print(f"[FIGHT] *** FAST-TRACK ***  diff>={F_FAST_TRACK_TH}  "
                          f"raw={res['raw']:.2f}", flush=True)
                elif res["detected"]:
                    self._consec_pos = min(6.0, self._consec_pos + 1)
                else:
                    self._consec_pos = max(0.0, self._consec_pos - 1)

                self._is_detected = (self._consec_pos >= F_CONSEC_MIN)

                tag = ("CONFIRMADO" if self._is_detected
                       else ("pico" if res["detected"] else "no"))
                print(f"[FIGHT] {tag}  smooth={res['score']:.2f}  raw={res['raw']:.2f}  "
                      f"hits={res['hits']}/{F_MIN_POS_FRAMES}  consec={self._consec_pos:.1f}",
                      flush=True)

                # Actualizar target-lock según score
                if self._last_score >= F_LOCK_SCORE_TH:
                    if not self._target_locked:
                        print(f"[FIGHT] TARGET_LOCK ON  ids={self._last_ids}  "
                              f"score={self._last_score:.2f}", flush=True)
                    self._target_locked = True
                elif self._last_score < 1.0:
                    if self._target_locked:
                        print(f"[FIGHT] TARGET_LOCK OFF  score={self._last_score:.2f}",
                              flush=True)
                    self._target_locked = False
                    self._locked_ids    = None
            except Exception as e:
                print(f"[FIGHT] Error worker: {e}", flush=True)
            self._processing = False

    # ------------------------------------------------------------------
    # Helpers espaciales
    # ------------------------------------------------------------------
    @staticmethod
    def _iou(b1, b2) -> float:
        xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
        xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter  = max(0.0, xB - xA) * max(0.0, yB - yA)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    @staticmethod
    def _center_dist(b1, b2) -> float:
        cx1, cy1 = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
        cx2, cy2 = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0
        return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))

    def _movement_score(self, p_id: int, box_p) -> float:
        prev = self._prev_boxes.get(p_id)
        if prev is None:
            return 0.0
        return (1.0 - self._iou(box_p, prev)) * 50.0

    def _score_pair(self, b1, b2, p_id1: int, p_id2: int) -> float:
        iou  = self._iou(b1, b2)
        dist = self._center_dist(b1, b2)
        prox = max(0.0, 1.0 - dist / F_DIST_MAX_LIMIT) * 40.0
        ov   = iou * 100.0
        mov  = (self._movement_score(p_id1, b1) + self._movement_score(p_id2, b2)) / 2.0
        return prox + ov + mov

    @staticmethod
    def _union_bbox(b1, b2, pad: int, W: int, H: int) -> Tuple[int, int, int, int]:
        x1 = int(max(0, min(b1[0], b2[0]) - pad))
        y1 = int(max(0, min(b1[1], b2[1]) - pad))
        x2 = int(min(W, max(b1[2], b2[2]) + pad))
        y2 = int(min(H, max(b1[3], b2[3]) + pad))
        return x1, y1, x2, y2

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------
    def _process_fight(self, frame: np.ndarray, person_xyxy, person_ids) -> None:
        n = len(person_xyxy)
        H, W = frame.shape[:2]

        # 1. Puntuar todos los pares dentro del rango
        candidatos = []
        for i in range(n):
            for j in range(i + 1, n):
                b1    = person_xyxy[i]
                b2    = person_xyxy[j]
                p_id1 = int(person_ids[i]) if i < len(person_ids) else -1
                p_id2 = int(person_ids[j]) if j < len(person_ids) else -1
                iou   = self._iou(b1, b2)
                dist  = self._center_dist(b1, b2)
                if dist < F_DIST_MAX_LIMIT or iou > F_MIN_IOU_CRITICAL:
                    sc = self._score_pair(b1, b2, p_id1, p_id2)
                    candidatos.append((sc, iou, i, j, p_id1, p_id2))

        # Actualizar historial de posiciones para scoring de movimiento
        ids_visibles = set()
        for i, box_p in enumerate(person_xyxy):
            p_id = int(person_ids[i]) if i < len(person_ids) else -1
            self._prev_boxes[p_id] = box_p.copy()
            ids_visibles.add(p_id)
        self._prev_boxes = {k: v for k, v in self._prev_boxes.items() if k in ids_visibles}

        # 2. Selección con target-lock
        best = None
        if candidatos:
            if self._target_locked and self._locked_ids is not None:
                lid1, lid2 = self._locked_ids
                locked = [c for c in candidatos
                          if (c[4] == lid1 and c[5] == lid2)
                          or (c[4] == lid2 and c[5] == lid1)]
                if locked:
                    best = max(locked, key=lambda x: x[0])
                else:
                    print(f"[FIGHT] TARGET_LOCK LOST  ids={self._locked_ids} no visibles",
                          flush=True)
                    self._target_locked = False
                    self._locked_ids    = None

            if best is None:
                best = max(candidatos, key=lambda x: x[0])

        # Actualizar locked_ids si hay candidato y score suficiente
        if best is not None and self._last_score >= F_LOCK_SCORE_TH:
            self._target_locked = True
            self._locked_ids    = (best[4], best[5])

        # 3. Generar crop del par activo
        p_crop = None

        if best is not None:
            sc, iou, i, j, p_id1, p_id2 = best
            b1, b2 = person_xyxy[i], person_xyxy[j]
            pad    = F_PAD_CLOSE if iou > 0.05 else F_PAD_PIXELS
            x1, y1, x2, y2 = self._union_bbox(b1, b2, pad, W, H)
            crop = frame[y1:y2, x1:x2]
            if crop is not None and crop.size > 0:
                p_crop = cv2.resize(crop, VIEW_SIZE)

            self._last_valid_bbox = (b1.copy(), b2.copy(), pad)
            self._sticky_counter  = F_STICKY_FRAMES
            self._last_ids        = (p_id1, p_id2)

            # Gate de movimiento: par casi estático → no acumular en CLIP
            mov1 = self._movement_score(p_id1, b1)
            mov2 = self._movement_score(p_id2, b2)
            pair_mov = (mov1 + mov2) / 2.0
            if pair_mov < F_MIN_MOVEMENT_FOR_CLIP:
                print(f"[FIGHT] MOV_GATE  ids=({p_id1},{p_id2})  "
                      f"mov={pair_mov:.2f}<{F_MIN_MOVEMENT_FOR_CLIP}  skip CLIP", flush=True)
                self._buffer = []
                return

        elif self._sticky_counter > 0 and self._last_valid_bbox is not None:
            print(f"[FIGHT] STICKY  counter={self._sticky_counter}", flush=True)
            b1, b2, pad = self._last_valid_bbox
            x1, y1, x2, y2 = self._union_bbox(b1, b2, pad, W, H)
            crop = frame[y1:y2, x1:x2]
            if crop is not None and crop.size > 0:
                p_crop = cv2.resize(crop, VIEW_SIZE)
            self._sticky_counter -= 1

        # 4. Buffer y despacho CLIP
        if p_crop is not None:
            self._last_crop = p_crop
            self._buffer.append(p_crop)
            if len(self._buffer) >= F_BUFFER_SIZE:
                if not self._processing:
                    try:
                        self._q.put_nowait(list(self._buffer))
                        self._processing = True
                        print(f"[FIGHT] CLIP_DISPATCH  frames={len(self._buffer)}  "
                              f"ids={self._last_ids}", flush=True)
                    except queue.Full:
                        print("[FIGHT] CLIP_DISPATCH SKIP  queue llena", flush=True)
                self._buffer = []
        else:
            self._buffer      = []
            self._consec_pos  = max(0.0, self._consec_pos - 0.2)
            self._is_detected = (self._consec_pos >= F_CONSEC_MIN)
