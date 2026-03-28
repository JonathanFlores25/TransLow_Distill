"""
core/tracker.py
===============
Encapsula YOLO + ByteTrack y devuelve un diccionario limpio con
personas y vehículos, listo para consumir por cualquier experto.
"""

import os

import numpy as np
from ultralytics import YOLO

# Clases COCO relevantes
_VEHICLE_CLASSES      = [2, 5, 7]      # car(2), bus(5), truck(7)  — para CrashExpert
_VEHICLE_CLASSES_ALL  = [2, 3, 5, 7]   # + motorcycle(3)           — para CarPartsExpert
_VEHICLE_MIN_CONF     = 0.45
_ALL_TRACK_CLASSES    = [0, 2, 3, 5, 7]


class ArconteTracker:
    """
    Wrapper de YOLO + ByteTrack.

    Uso:
        tracker = ArconteTracker()
        tracker_data = tracker.process(frame)

    tracker_data keys:
        "persons_xyxy"    np.ndarray [N,4]  float32
        "persons_ids"     np.ndarray [N]    int32
        "vehicles_xyxy"   np.ndarray [M,4]  float32
        "vehicles_ids"    np.ndarray [M]    int32
        "vehicles_confs"  np.ndarray [M]    float32
        "frame_idx"       int
    """

    def __init__(
        self,
        model_path: str   = ".checkpoints/yolo11x.pt",
        tracker_yaml: str = "bytetrack.yaml",
    ):
        engine = model_path.replace(".pt", ".engine")
        self._model        = YOLO(engine if os.path.exists(engine) else model_path)
        self._tracker_yaml = tracker_yaml
        self._frame_idx    = 0

    def process(self, frame: np.ndarray) -> dict:
        res = self._model.track(
            frame,
            classes=_ALL_TRACK_CLASSES,
            persist=True,
            verbose=False,
            tracker=self._tracker_yaml,
        )[0]

        if res.boxes is None or len(res.boxes) == 0:
            all_xyxy  = np.zeros((0, 4), dtype=np.float32)
            all_ids   = np.zeros((0,),   dtype=np.int32)
            all_cls   = np.zeros((0,),   dtype=np.int32)
            all_conf  = np.zeros((0,),   dtype=np.float32)
        else:
            b        = res.boxes
            all_xyxy = b.xyxy.cpu().numpy().astype(np.float32)
            all_ids  = (
                b.id.cpu().numpy().astype(np.int32)
                if b.id is not None
                else np.full(len(all_xyxy), -1, dtype=np.int32)
            )
            all_cls  = b.cls.cpu().numpy().astype(np.int32)
            all_conf = b.conf.cpu().numpy().astype(np.float32)

        # Personas (clase 0)
        pm           = all_cls == 0
        person_xyxy  = all_xyxy[pm]
        person_ids   = all_ids[pm]

        # Vehículos sin motos (clases 2,5,7) — para CrashExpert
        vm            = np.isin(all_cls, _VEHICLE_CLASSES) & (all_conf >= _VEHICLE_MIN_CONF)
        vehicle_xyxy  = all_xyxy[vm]
        vehicle_ids   = all_ids[vm]
        vehicle_confs = all_conf[vm]

        # Vehículos con motos (clases 2,3,5,7) — para CarPartsExpert
        vm_all            = np.isin(all_cls, _VEHICLE_CLASSES_ALL) & (all_conf >= _VEHICLE_MIN_CONF)
        vehicle_all_xyxy  = all_xyxy[vm_all]
        vehicle_all_ids   = all_ids[vm_all]

        self._frame_idx += 1

        return {
            "persons_xyxy":    person_xyxy,
            "persons_ids":     person_ids,
            "vehicles_xyxy":   vehicle_xyxy,    # sin motos — CrashExpert
            "vehicles_ids":    vehicle_ids,
            "vehicles_confs":  vehicle_confs,
            "vehicles_all_xyxy": vehicle_all_xyxy,  # con motos — CarPartsExpert
            "vehicles_all_ids":  vehicle_all_ids,
            "frame_idx":       self._frame_idx,
        }
