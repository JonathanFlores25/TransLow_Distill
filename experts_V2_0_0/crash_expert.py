"""
experts/crash_expert.py
=======================
Experto de detección de CHOQUES.
Lógica idéntica a ArconteV3Prod._process_crash + CLIPCrashExpert.
"""

import queue
import threading
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import clip

from core.base_expert import BaseExpert

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
clip_device = "cuda" if torch.cuda.is_available() else "cpu"
VIEW_SIZE   = (640, 480)

# ---------------------------------------------------------------------------
# Constantes CRASH (V7: calibradas desde debug_crash_only.py)
# ---------------------------------------------------------------------------
C_CAR_PAD_PX            = 75
C_TRIGGER_FRAMES        = 4
C_BUFFER_SIZE           = 64
C_MOTION_THRESHOLD      = 10
C_CAR_PAIR_IOU_THR      = 0.30
C_STATIONARY_THR_FRAMES = 15
C_REAR_ANGLE_TOL        = 35.0
C_REAR_DIST_MULT        = 1.0
C_POS_THRESHOLD         = 0.90
C_MIN_POS_FRAMES        = 2
C_MIN_SET_SCORE         = 0.90
C_MARGIN_MIN            = 0.50
C_TOPK                  = 3
C_CLIP_STRIDE           = 16
C_CRASH_TTL             = 0
C_CRASH_MAX_FRAMES      = 90   # V8: cap duro
C_CONSEC_MIN            = 2
# Anti-parking
C_CLOSING_HISTORY       = 5
C_CLOSING_DROP_RATIO    = 0.10
C_MOTION_HISTORY_LEN    = 5
C_MOTION_SUSTAINED_MIN  = 2
C_CONSEC_RUN_MIN        = 2

# ---------------------------------------------------------------------------
# Prompts CLIP
# ---------------------------------------------------------------------------
CRASH_POS_PROMPTS = [
    "a traffic accident with visibly damaged cars",
    "a wrecked car with a smashed front bumper",
    "crumpled car metal after a collision",
    "broken glass and car parts scattered on the road after a crash",
    "an accident scene with stopped cars and debris",
    "a car crash with airbags deployed",
    "a smashed car with a broken windshield after a crash",
    "two cars rear-end collision at an intersection",
    "a traffic accident with two damaged cars after a collision",
    "a car crash scene with debris and broken glass on the road",
    "a serious vehicle collision with crumpled metal and damaged cars",
    "an accident scene with crashed vehicles blocking the road",
    "cars badly damaged after a traffic accident",
    "a car crash with airbags deployed and front-end damage",
    "two moving cars colliding at an intersection",
    "a side-impact collision between two driving cars",
    "a car crash where two vehicles hit each other while driving",
    "two cars crashing into each other on a city street",
    "a moving car crashing into another moving vehicle",
    "a high-speed collision between two cars on the road",
    "two vehicles colliding while driving through an intersection",
    "cars crashing together in the middle of traffic",
    "a moving car crashing into a parked car",
    "a driving car hitting a stationary vehicle on the roadside",
    "a car rear-ending a stopped car in traffic",
    "a moving vehicle colliding with a parked car on the street",
    "a car crashing into a stationary car at the curb",
    "a driving car hitting a stopped vehicle at a traffic light",
    "a car accident where a moving vehicle hits a parked car",
    "a car crashing into a tree beside the road",
    "a vehicle hitting a street pole or traffic light",
    "a car crashed into a wall with a damaged front",
    "a car that has driven into a guardrail",
    "a vehicle crashing into a concrete barrier",
    "a car hitting a roadside fence",
    "a car that crashed into a building or storefront",
    "a vehicle that drove onto the sidewalk and hit a pole",
    "a car with a smashed front end after a collision",
    "a vehicle with severe crash damage and broken windshield",
    "a wrecked car with crumpled metal and broken parts",
    "a car badly damaged after hitting something",
]

CRASH_NEG_PROMPTS = [
    "a normal street scene with buildings and cars",
    "an ordinary urban scene with no emergency",
    "a calm daytime street scene with no emergency",
    "cars driving normally on a road with no accident",
    "normal traffic flowing on a city street",
    "vehicles moving normally on a highway with no collision",
    "a road scene with cars and no collision",
    "a parking lot with cars parked normally",
    "people fighting and throwing punches on the street",
    "cars driving normally on a road with no accident",
    "normal traffic moving through an intersection",
    "vehicles driving safely on a city street",
    "cars parked neatly along the side of the road",
    "a calm urban street with normal traffic",
    "a highway scene with vehicles moving normally",
    "a parking lot full of parked cars with no damage",
    "cars waiting at a traffic light with no collision",
    "a quiet residential street with parked cars",
    "a normal road scene with vehicles and no accident",
    "a street scene with people walking and cars driving normally",
    "a busy road with traffic flowing normally",
    "a car stopped legally at the curb",
    "a road with vehicles but no crash or collision",
    "people arguing or fighting on the street",
    "a car turning at an intersection with no accident",
    "a car turning into a driveway with no collision",
    "a car turning around a corner and leaving the frame with no accident",
    "a car slowly parking into a parking spot next to another car",
    "a car parallel parking carefully near other vehicles",
    "a vehicle reversing slowly into a parking space",
    "a car maneuvering into a tight parking spot without hitting anything",
    "two cars parked very close together with no damage",
    "a car slowly approaching another car in heavy traffic without collision",
    "a vehicle inching forward in stop-and-go traffic",
    "a car backing up slowly near a parked vehicle",
    "cars very close to each other in a crowded parking lot with no accident",
    "a car pulling into a driveway slowly next to another parked car",
]


class CrashExpert(BaseExpert):
    """
    Detecta choques de vehículos usando:
      1. Heurística espacial: distancia entre vehículos + closing-speed gate.
      2. Buffer de C_BUFFER_SIZE crops (64 frames) enviado al worker CLIP.
      3. Confirmación CLIP con stride=16, softmax pos/neg.
      4. Hard cap: C_CRASH_MAX_FRAMES (90 frames).
    """

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------
    def __init__(self):
        # Estado CRASH
        self.crash_counter           = 0
        self.crash_locked            = False
        self.crash_buffer: list      = []
        self.crash_is_detected: bool = False
        self.crash_ttl_counter       = 0
        self.crash_consec_pos        = 0
        self.crash_last_crop         = None
        self.crash_car_prev_center: dict = {}
        self._crash_processing: bool = False
        self.crash_interaction_type  = None
        self.active_car_pair         = None
        self.crashed_stationary_cars: list = []
        self.active_pileup           = None
        self._prev_car_boxes: list   = []
        self.candidate_sent_count    = 0
        self.crash_ever_detected     = False
        self.crash_active_frames     = 0   # V8: cap duro
        # Anti-parking state
        self._motion_history: dict   = {}
        self._pair_dist_history: dict = {}

        # CLIP
        self._model        = None
        self._preprocess   = None
        self._text_features = None

        # Queue del worker
        self._q: queue.Queue = queue.Queue(maxsize=2)

    # ------------------------------------------------------------------
    # BaseExpert — propiedades
    # ------------------------------------------------------------------
    @property
    def label(self) -> str:
        return "CHOQUE"

    @property
    def is_active(self) -> bool:
        return self.crash_is_detected

    # ------------------------------------------------------------------
    # BaseExpert — load
    # ------------------------------------------------------------------
    def load(self, model, preprocess) -> None:
        self._model      = model
        self._preprocess = preprocess

        with torch.no_grad():
            pt = clip.tokenize(CRASH_POS_PROMPTS).to(clip_device)
            nt = clip.tokenize(CRASH_NEG_PROMPTS).to(clip_device)
            pm = model.encode_text(pt).mean(0, keepdim=True)
            nm = model.encode_text(nt).mean(0, keepdim=True)
            pm /= pm.norm(dim=-1, keepdim=True)
            nm /= nm.norm(dim=-1, keepdim=True)
            self._text_features = torch.cat([pm, nm], dim=0)

        threading.Thread(target=self._worker_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # BaseExpert — process_heuristics
    # ------------------------------------------------------------------
    def process_heuristics(self, frame: np.ndarray, tracker_data: Dict[str, Any]) -> None:
        car_xyxy    = tracker_data["vehicles_xyxy"]
        car_ids     = tracker_data["vehicles_ids"]
        person_xyxy = tracker_data["persons_xyxy"]

        # V8: cap duro de duración para choque
        if self.crash_is_detected:
            self.crash_active_frames += 1
            if self.crash_active_frames > C_CRASH_MAX_FRAMES:
                self.crash_is_detected   = False
                self.crash_active_frames = 0
                self.crash_consec_pos    = 0
        else:
            self.crash_active_frames = 0

        self._process_crash(frame, car_xyxy, car_ids, person_xyxy)

    # ------------------------------------------------------------------
    # BaseExpert — predict
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, frames: List[np.ndarray]) -> dict:
        sampled = frames[::max(1, C_CLIP_STRIDE)]
        valid   = [fr for fr in sampled if fr is not None and fr.size != 0]
        if not valid:
            return {"detected": False, "set_score": 0.0, "hits": 0, "per_frame": []}

        imgs = torch.stack([
            self._preprocess(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            for fr in valid
        ]).to(clip_device)

        feats = self._model.encode_image(imgs)
        feats /= feats.norm(dim=-1, keepdim=True)
        sim   = self._model.logit_scale.exp() * (feats @ self._text_features.T)
        probs = sim.softmax(dim=-1).cpu().numpy()

        pa = probs[:, 0].astype(np.float32)
        na = probs[:, 1].astype(np.float32)
        ma = pa - na

        per_frame_scores = [
            {"pos": float(pa[i]), "neg": float(na[i]), "margin": float(ma[i])}
            for i in range(len(pa))
        ]

        hit_mask = (pa > C_POS_THRESHOLD) & (ma > C_MARGIN_MIN)
        hits     = int(hit_mask.sum())
        good     = pa[hit_mask]

        if good.size >= C_TOPK:
            set_score = float(np.sort(good)[-C_TOPK:].mean())
        elif good.size > 0:
            set_score = float(good.mean())
        else:
            set_score = float(pa.mean())

        detected = bool(hits >= C_MIN_POS_FRAMES and set_score >= C_MIN_SET_SCORE)
        return {
            "detected":  detected,
            "score":     round(set_score, 3),
            "set_score": round(set_score, 3),
            "hits":      hits,
            "per_frame": per_frame_scores,
        }

    # ------------------------------------------------------------------
    # BaseExpert — get_display_data
    # ------------------------------------------------------------------
    def get_display_data(self) -> Dict[str, Any]:
        return {"last_crop": self.crash_last_crop} if self.crash_last_crop is not None else {}

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            payload = self._q.get()
            frames  = payload["frames"]
            try:
                res                    = self.predict(frames)
                total_positive         = res["hits"]
                per_frame              = res.get("per_frame", [])
                max_run, cur_run       = 0, 0
                for pf in per_frame:
                    if pf["pos"] > C_POS_THRESHOLD and pf["margin"] > C_MARGIN_MIN:
                        cur_run += 1
                        max_run  = max(max_run, cur_run)
                    else:
                        cur_run = 0

                confirmed = (total_positive >= C_CONSEC_MIN and max_run >= C_CONSEC_RUN_MIN)
                if confirmed:
                    self.crash_consec_pos    = total_positive
                    self.crash_is_detected   = True
                    self.crash_ttl_counter   = 0
                    self.crash_ever_detected = True
                else:
                    self.crash_consec_pos = total_positive
                    if self.crash_is_detected:
                        self.crash_ttl_counter += 1
                        if self.crash_ttl_counter > C_CRASH_TTL:
                            self.crash_is_detected = False
                            self.crash_consec_pos  = 0
            except Exception:
                pass
            self._crash_processing = False

    # ------------------------------------------------------------------
    # Helpers (idénticos a ArconteV3Prod)
    # ------------------------------------------------------------------
    @staticmethod
    def _xyxy_center(b):
        return np.array([(b[0]+b[2])/2.0, (b[1]+b[3])/2.0], dtype=np.float32)

    @staticmethod
    def _iou_xyxy(bA, bB) -> float:
        xA1, yA1, xA2, yA2 = float(bA[0]), float(bA[1]), float(bA[2]), float(bA[3])
        xB1, yB1, xB2, yB2 = float(bB[0]), float(bB[1]), float(bB[2]), float(bB[3])
        inter_x1 = max(xA1, xB1); inter_y1 = max(yA1, yB1)
        inter_x2 = min(xA2, xB2); inter_y2 = min(yA2, yB2)
        inter    = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        areaA    = max(0.0, xA2 - xA1) * max(0.0, yA2 - yA1)
        areaB    = max(0.0, xB2 - xB1) * max(0.0, yB2 - yB1)
        union    = areaA + areaB - inter
        return float(inter / union) if union > 0.0 else 0.0

    def _match_box_by_iou(self, prev_box, current_boxes, iou_thr):
        best_box, best_iou = None, 0.0
        for b in current_boxes:
            iou = self._iou_xyxy(prev_box, b)
            if iou > best_iou:
                best_iou = iou
                best_box = b
        if best_box is None or best_iou < iou_thr:
            return None, best_iou
        return best_box, best_iou

    @staticmethod
    def _crop_union_with_padding(frame, b1, b2, pad_px) -> Optional[np.ndarray]:
        x1 = min(float(b1[0]), float(b2[0]))
        y1 = min(float(b1[1]), float(b2[1]))
        x2 = max(float(b1[2]), float(b2[2]))
        y2 = max(float(b1[3]), float(b2[3]))
        H, W = frame.shape[:2]
        x1, y1 = max(0, int(x1 - pad_px)), max(0, int(y1 - pad_px))
        x2, y2 = min(W - 1, int(x2 + pad_px)), min(H - 1, int(y2 + pad_px))
        if x2 < x1 or y2 < y1:
            return None
        crop = frame[y1:y2 + 1, x1:x2 + 1]
        return cv2.resize(crop, VIEW_SIZE) if crop.size > 0 else None

    @staticmethod
    def _crop_multi_car_union(frame, boxes, pad_px) -> Optional[np.ndarray]:
        if not boxes:
            return None
        x1 = min(float(b[0]) for b in boxes)
        y1 = min(float(b[1]) for b in boxes)
        x2 = max(float(b[2]) for b in boxes)
        y2 = max(float(b[3]) for b in boxes)
        H, W = frame.shape[:2]
        x1, y1 = max(0, int(x1 - pad_px)), max(0, int(y1 - pad_px))
        x2, y2 = min(W - 1, int(x2 + pad_px)), min(H - 1, int(y2 + pad_px))
        if x2 < x1 or y2 < y1:
            return None
        crop = frame[y1:y2 + 1, x1:x2 + 1]
        return cv2.resize(crop, VIEW_SIZE) if crop.size > 0 else None

    @staticmethod
    def _short_side_segments(box):
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        w, h = x2 - x1, y2 - y1
        if w <= h:
            a1, b1 = np.array([x1, y1], dtype=float), np.array([x1, y2], dtype=float)
            a2, b2 = np.array([x2, y1], dtype=float), np.array([x2, y2], dtype=float)
        else:
            a1, b1 = np.array([x1, y1], dtype=float), np.array([x2, y1], dtype=float)
            a2, b2 = np.array([x1, y2], dtype=float), np.array([x2, y2], dtype=float)
        return [(a1, b1), (a2, b2)]

    @staticmethod
    def _point_to_segment_distance(p, a, b) -> float:
        p  = np.asarray(p, dtype=float)
        a  = np.asarray(a, dtype=float)
        b  = np.asarray(b, dtype=float)
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 <= 1e-12:
            return float(np.linalg.norm(p - a))
        t = float(np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0))
        return float(np.linalg.norm(p - (a + t * ab)))

    @staticmethod
    def _adaptive_pair_threshold(b1, b2, base_px=85.0, k=0.35, min_px=120.0, max_px=600.0) -> float:
        d1 = np.hypot(b1[2] - b1[0], b1[3] - b1[1])
        d2 = np.hypot(b2[2] - b2[0], b2[3] - b2[1])
        return float(np.clip(base_px + k * 0.5 * (d1 + d2), min_px, max_px))

    def _car_is_moving(self, car_id: int, center) -> bool:
        prev = self.crash_car_prev_center.get(car_id)
        self.crash_car_prev_center[car_id] = center
        if prev is None:
            moved_now = True
            inst_dist = 0.0
        else:
            inst_dist = float(np.linalg.norm(center - prev))
            moved_now = inst_dist > C_MOTION_THRESHOLD
        if car_id not in self._motion_history:
            self._motion_history[car_id] = deque(maxlen=C_MOTION_HISTORY_LEN)
        self._motion_history[car_id].append(moved_now)
        if inst_dist > C_MOTION_THRESHOLD * 3:
            return True
        hist = self._motion_history[car_id]
        return sum(hist) >= C_MOTION_SUSTAINED_MIN

    def _is_closing_fast(self, id1: int, id2: int, current_dist: float) -> bool:
        key = (min(id1, id2), max(id1, id2))
        if key not in self._pair_dist_history:
            self._pair_dist_history[key] = deque(maxlen=C_CLOSING_HISTORY)
        hist = self._pair_dist_history[key]
        hist.append(current_dist)
        if len(hist) < 2:
            return True
        oldest = hist[0]
        if oldest < 1e-6:
            return True
        drop = (oldest - current_dist) / oldest
        return drop >= C_CLOSING_DROP_RATIO

    def _is_rear_collision(self, moving_car, stationary_car) -> bool:
        mc_cx = (moving_car[0]     + moving_car[2])     / 2.0
        mc_cy = (moving_car[1]     + moving_car[3])     / 2.0
        sc_cx = (stationary_car[0] + stationary_car[2]) / 2.0
        sc_cy = (stationary_car[1] + stationary_car[3]) / 2.0
        angle    = np.abs(np.degrees(np.arctan2(mc_cy - sc_cy, mc_cx - sc_cx)))
        is_horiz = angle < C_REAR_ANGLE_TOL or angle > (180.0 - C_REAR_ANGLE_TOL)
        is_vert  = (90.0 - C_REAR_ANGLE_TOL) < angle < (90.0 + C_REAR_ANGLE_TOL)
        return is_horiz or is_vert

    def _update_crashed_cars(self, car_boxes) -> None:
        updated = []
        for cc in self.crashed_stationary_cars:
            match, iou = self._match_box_by_iou(cc["box"], car_boxes, 0.4)
            if match is not None and iou > 0.7:
                updated.append({
                    "box":               np.array(match[:4], dtype=float),
                    "frames_stationary": cc["frames_stationary"] + 1,
                })
        self.crashed_stationary_cars = updated

    def _detect_pileup_collision(
        self, frame: np.ndarray, car_boxes
    ) -> Tuple[bool, Optional[np.ndarray], Optional[int]]:
        if not self.crashed_stationary_cars:
            return False, None, None
        targets = [c for c in self.crashed_stationary_cars
                   if c["frames_stationary"] >= C_STATIONARY_THR_FRAMES]
        if not targets:
            return False, None, None
        for car in car_boxes:
            car_center = np.array([(car[0]+car[2])/2.0, (car[1]+car[3])/2.0])
            for tgt in targets:
                tgt_box    = tgt["box"]
                tgt_center = np.array([(tgt_box[0]+tgt_box[2])/2.0, (tgt_box[1]+tgt_box[3])/2.0])
                if self._iou_xyxy(car, tgt_box) > 0.5:
                    continue
                segs_car = self._short_side_segments(car)
                segs_tgt = self._short_side_segments(tgt_box)
                d11  = self._point_to_segment_distance(tgt_center, segs_car[0][0], segs_car[0][1])
                d12  = self._point_to_segment_distance(tgt_center, segs_car[1][0], segs_car[1][1])
                d21  = self._point_to_segment_distance(car_center, segs_tgt[0][0], segs_tgt[0][1])
                d22  = self._point_to_segment_distance(car_center, segs_tgt[1][0], segs_tgt[1][1])
                dist = min(d11, d12, d21, d22)
                thr  = self._adaptive_pair_threshold(car, tgt_box) * C_REAR_DIST_MULT
                if dist < thr and self._is_rear_collision(car, tgt_box):
                    involved = [tgt_box, car]
                    for other in targets:
                        if np.array_equal(other["box"], tgt_box):
                            continue
                        oc = np.array([(other["box"][0]+other["box"][2])/2.0,
                                       (other["box"][1]+other["box"][3])/2.0])
                        d_to_primary  = np.linalg.norm(tgt_center - oc)
                        combined_diag = (
                            np.hypot(tgt_box[2]-tgt_box[0], tgt_box[3]-tgt_box[1]) +
                            np.hypot(other["box"][2]-other["box"][0], other["box"][3]-other["box"][1])
                        ) / 2.0
                        if d_to_primary < combined_diag * 2.5:
                            involved.append(other["box"])
                    crop = self._crop_multi_car_union(frame, involved, C_CAR_PAD_PX)
                    return True, crop, len(involved)
        return False, None, None

    def _get_car_car_crop(
        self, frame: np.ndarray, car_xyxy, car_ids
    ) -> Tuple[bool, Optional[np.ndarray], Optional[tuple]]:
        for i in range(len(car_xyxy)):
            for j in range(i + 1, len(car_xyxy)):
                b1, b2 = car_xyxy[i], car_xyxy[j]
                c1     = self._xyxy_center(b1)
                c2     = self._xyxy_center(b2)
                id1    = int(car_ids[i]) if car_ids[i] >= 0 else i
                id2    = int(car_ids[j]) if car_ids[j] >= 0 else j
                if not self._car_is_moving(id1, c1):
                    continue
                segs1 = self._short_side_segments(b1)
                segs2 = self._short_side_segments(b2)
                d11   = self._point_to_segment_distance(c2, segs1[0][0], segs1[0][1])
                d12   = self._point_to_segment_distance(c2, segs1[1][0], segs1[1][1])
                d21   = self._point_to_segment_distance(c1, segs2[0][0], segs2[0][1])
                d22   = self._point_to_segment_distance(c1, segs2[1][0], segs2[1][1])
                dist  = min(d11, d12, d21, d22)
                thr   = self._adaptive_pair_threshold(b1, b2)
                if dist < thr:
                    if not self._is_closing_fast(id1, id2, dist):
                        continue
                    crop = self._crop_union_with_padding(frame, b1, b2, C_CAR_PAD_PX)
                    if crop is not None:
                        return True, crop, (b1, b2)
        return False, None, None

    # ------------------------------------------------------------------
    # Pipeline principal (idéntico a ArconteV3Prod._process_crash)
    # ------------------------------------------------------------------
    def _process_crash(self, frame: np.ndarray, car_xyxy, car_ids, person_xyxy) -> None:
        c_crop = None

        if len(car_xyxy) > 0:
            self._update_crashed_cars(car_xyxy)

        if not self.crash_locked:
            # Prioridad 1: pile-up
            pu_found, pu_crop, pu_num = self._detect_pileup_collision(frame, car_xyxy)
            if pu_found and pu_crop is not None:
                c_crop                 = pu_crop
                self.crash_last_crop   = pu_crop
                self.crash_counter    += 1
                if self.crash_counter >= C_TRIGGER_FRAMES:
                    self.crash_locked             = True
                    self.crash_interaction_type   = "multi-car"
                    self.active_pileup            = {
                        "cars":               [np.array(b[:4], dtype=float) for b in car_xyxy],
                        "expansion_counter":  0,
                    }
                    self.active_car_pair          = None
                    self.crash_buffer.append(pu_crop)
            else:
                # Prioridad 2: car-car
                cc_found, cc_crop, cc_pair = self._get_car_car_crop(frame, car_xyxy, car_ids)
                if cc_found and cc_crop is not None:
                    c_crop               = cc_crop
                    self.crash_last_crop = cc_crop
                    self.crash_counter  += 1
                    if self.crash_counter >= C_TRIGGER_FRAMES:
                        self.crash_locked           = True
                        self.crash_interaction_type = "car-car"
                        self.active_pileup          = None
                        if cc_pair is not None:
                            b1, b2 = cc_pair
                            self.active_car_pair = {
                                "b1": np.array(b1[:4], dtype=float),
                                "b2": np.array(b2[:4], dtype=float),
                            }
                            self.crashed_stationary_cars.append(
                                {"box": np.array(b1[:4], dtype=float), "frames_stationary": 0}
                            )
                            self.crashed_stationary_cars.append(
                                {"box": np.array(b2[:4], dtype=float), "frames_stationary": 0}
                            )
                        else:
                            self.active_car_pair = None
                        self.crash_buffer.append(cc_crop)
                else:
                    self.crash_counter = max(0, self.crash_counter - 1.5)
                    if self.crash_counter <= 0:
                        self.crash_buffer           = []
                        self.active_car_pair        = None
                        self.active_pileup          = None
                        self.crash_interaction_type = None

        else:
            # LOCKED: estabilización por IoU
            sample = self.crash_last_crop

            if self.crash_interaction_type == "multi-car":
                if self.active_pileup is not None and len(car_xyxy) > 0:
                    matched = []
                    for pc in self.active_pileup["cars"]:
                        m, _ = self._match_box_by_iou(pc, car_xyxy, 0.3)
                        if m is not None:
                            matched.append(np.array(m[:4], dtype=float))
                    if matched:
                        self.active_pileup["cars"] = matched
                        crop = self._crop_multi_car_union(frame, matched, C_CAR_PAD_PX)
                        if crop is not None:
                            sample               = crop
                            self.crash_last_crop = crop

            elif self.crash_interaction_type == "car-car":
                if self.active_car_pair is not None and len(car_xyxy) > 0:
                    prev_b1   = self.active_car_pair["b1"]
                    prev_b2   = self.active_car_pair["b2"]
                    m1, _     = self._match_box_by_iou(prev_b1, car_xyxy, C_CAR_PAIR_IOU_THR)
                    remaining = list(car_xyxy)
                    if m1 is not None:
                        filtered = []
                        removed  = False
                        for b in remaining:
                            if (not removed) and np.allclose(
                                np.array(b[:4], dtype=float),
                                np.array(m1[:4], dtype=float),
                            ):
                                removed = True
                                continue
                            filtered.append(b)
                        remaining = filtered
                    m2, _ = self._match_box_by_iou(prev_b2, remaining, C_CAR_PAIR_IOU_THR)
                    if m1 is not None and m2 is not None:
                        self.active_car_pair = {
                            "b1": np.array(m1[:4], dtype=float),
                            "b2": np.array(m2[:4], dtype=float),
                        }
                        crop = self._crop_union_with_padding(frame, m1, m2, C_CAR_PAD_PX)
                        if crop is not None:
                            sample               = crop
                            self.crash_last_crop = crop

            if sample is not None:
                c_crop = sample
                self.crash_buffer.append(sample)

            # Buffer lleno → despachar a CLIP → desbloquear
            if len(self.crash_buffer) >= C_BUFFER_SIZE:
                self.candidate_sent_count += 1
                if not self._crash_processing:
                    try:
                        self._q.put_nowait({
                            "frames":      list(self.crash_buffer),
                            "dispatch_id": self.candidate_sent_count,
                        })
                        self._crash_processing = True
                    except queue.Full:
                        self._crash_processing = False
                self.crash_locked           = False
                self.crash_counter          = 0
                self.crash_buffer           = []
                self.active_car_pair        = None
                self.active_pileup          = None
                self.crash_interaction_type = None

        self._prev_car_boxes = [np.array(b[:4], dtype=float) for b in car_xyxy]
