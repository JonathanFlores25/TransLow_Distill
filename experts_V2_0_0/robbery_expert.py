"""
experts/robbery_expert.py
=========================
Experto de detección de ROBO CON VIOLENCIA.

Pipeline
--------
1. process_heuristics() recibe frame + tracker_data del orquestador compartido.
2. PersonEventDetector evalúa cada par de personas en pixel-space:
       - Puerta de proximidad  : distancia < PROX_DISTANCE_PX
       - Velocidad de cierre   : distancia cae >= CLOSING_DROP_RATIO en ventana
       - Movimiento asimétrico : un agente se mueve significativamente más rápido
3. Al disparar un evento, el crop del par se calcula y se programa para inferencia
   POST_FRAMES frames más adelante (para tener el clip completo).
4. Cuando llega el frame ready_at, se ensambla un clip de NUM_SEGMENTS frames
   y se envía al worker thread via queue.
5. El worker thread corre ActionCLIP sobre el clip.
6. Confirmación: >= CONSEC_CLIPS_REQUIRED clips positivos consecutivos.
7. TTL: la etiqueta ROBO permanece ROBBERY_TTL frames tras la confirmación.

Integración con el framework ZeroShot
--------------------------------------
- Hereda de BaseExpert.
- load(model, preprocess) acepta el CLIP compartido pero NO lo usa:
  ActionCLIP carga su propio backbone y cabeza temporal desde un checkpoint.
- El tracker compartido (ArconteTracker) ya devuelve persons_xyxy / persons_ids
  (clase 0); este experto los convierte internamente al formato de dicts que
  necesita PersonEventDetector.
- El frame buffer es interno (deque de FrameEntry); se alimenta desde el
  argumento `frame` de process_heuristics(), igual que hacen los demás expertos.

Dependencias externas
---------------------
  ActionCLIP repo en PYTHONPATH o en ACTIONCLIP_PATH:
      clip                        (fork modificado incluido en el repo)
      modules.Visual_Prompt       (cabeza de fusión temporal)
  torch, torchvision, pillow, numpy, opencv-python
"""

import os
import sys
import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from core.base_expert import BaseExpert

# ---------------------------------------------------------------------------
# ActionCLIP repo path — bundled inside the ZeroShot repo at ActionCLIP/
# ---------------------------------------------------------------------------
_ACTIONCLIP_REPO = os.environ.get(
    "ACTIONCLIP_PATH",
    os.path.join(os.path.dirname(__file__), "..", "ActionCLIP"),
)
_ACTIONCLIP_REPO = os.path.abspath(_ACTIONCLIP_REPO)
if _ACTIONCLIP_REPO not in sys.path:
    sys.path.insert(0, _ACTIONCLIP_REPO)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Prompts ActionCLIP
# ---------------------------------------------------------------------------
_POS_PROMPTS = [
    # Fase de aproximación y amenaza
    "a person aggressively approaching another person on a street",
    "a person cornering another person against a wall",
    "a person threatening another with a weapon on the street",
    "a person pointing a knife or gun at another person",
    "a mugger grabbing a victim from behind on a sidewalk",
    # Fase de contacto y sustracción
    "a person violently grabbing a bag or phone from another person",
    "a mugging where an aggressor pushes a victim to the ground",
    "a person snatching an object from someone and running away",
    "a violent street robbery with physical aggression",
    "an aggressor forcing a victim against a wall and taking their belongings",
    # Framing CCTV
    "a CCTV recording of a street robbery",
    "an overhead security camera recording of a person being robbed",
    "a close-up of two people where one is threatening the other",
    "a person grabbing another person at close range on a sidewalk",
]

_NEG_PROMPTS = [
    "two people shaking hands or greeting each other on a street",
    "a person handing something to another person voluntarily",
    "two people hugging on a sidewalk",
    "two friends talking closely face to face",
    "a person helping another who has fallen on the ground",
    "a person passing a bag to a friend",
    "people walking normally on a sidewalk",
    "a person jogging on a street with no one else near",
    "two people walking side by side at normal pace",
    "a person asking another for directions on the street",
    "people standing in a group having a conversation",
    "people waiting at a bus stop standing close together",
    "a crowd of pedestrians walking through a busy area",
    "two people standing close together with no aggression",
    "a security camera view of people walking peacefully",
]

# ---------------------------------------------------------------------------
# Parámetros de detección
# ---------------------------------------------------------------------------
# ActionCLIP
NUM_SEGMENTS          = 32     # frames enviados a ActionCLIP
PRE_FRAMES            = 20     # frames antes del evento
POST_FRAMES           = NUM_SEGMENTS - PRE_FRAMES - 1   # = 11
POS_THRESHOLD         = 0.50   # softmax pos_prob mínimo
MIN_MARGIN            = 0.05   # pos_prob - neg_prob mínimo
CONSEC_CLIPS_REQUIRED = 2      # clips positivos consecutivos para confirmar

# Heurística pixel-space
PROX_DISTANCE_PX    = 250      # distancia máxima de centroides para formar par
CLOSING_HISTORY_LEN = 5        # frames de historial de distancia por par
CLOSING_DROP_RATIO  = 1.0      # caída relativa mínima en la ventana
MOTION_HISTORY_LEN  = 5        # frames de historial de velocidad por persona
MOTION_THR_PX       = 6        # px/frame para considerar que una persona se mueve
ASYMMETRY_RATIO     = 0.5      # (fast-slow)/fast mínimo para "movimiento asimétrico"
EVENT_COOLDOWN      = 60       # frames de cooldown entre eventos del mismo par

# Crop y TTL
CROP_PAD_PX   = 120            # padding alrededor del par en el crop
ROBBERY_TTL   = 60             # frames con etiqueta ROBO tras confirmación

# Buffer de frames interno (capacidad holgada)
_BUF_CAPACITY = (PRE_FRAMES + POST_FRAMES + 10) * 2 + 60


# ---------------------------------------------------------------------------
# Estructuras de datos internas
# ---------------------------------------------------------------------------

@dataclass
class _FrameEntry:
    """Entrada del buffer de frames interno."""
    image:        np.ndarray
    frame_idx:    int


@dataclass
class _PersonCandidateEvent:
    """Evento candidato emitido por _PersonEventDetector."""
    frame_idx: int
    pair_key:  Tuple[int, int]   # (id_a, id_b) ordenados
    relation:  str               # "closing" | "contact" | "asymmetric"
    score:     float             # confianza heurística [0, 1]


# ---------------------------------------------------------------------------
# Detector de eventos pixel-space
# ---------------------------------------------------------------------------

class _PersonEventDetector:
    """
    Evalúa cada par de personas visibles en un frame y emite
    _PersonCandidateEvent cuando se cumple la puerta de proximidad
    más al menos una señal (cierre o asimetría).
    """

    def __init__(self):
        self._speed_history:    Dict[int, deque] = {}
        self._prev_center:      Dict[int, np.ndarray] = {}
        self._dist_history:     Dict[Tuple[int, int], deque] = {}
        self._last_event_frame: Dict[Tuple[int, int], int] = {}

    def _update_speed(self, tid: int, center: np.ndarray) -> float:
        prev = self._prev_center.get(tid)
        self._prev_center[tid] = center.copy()
        inst = 0.0 if prev is None else float(np.linalg.norm(center - prev))
        if tid not in self._speed_history:
            self._speed_history[tid] = deque(maxlen=MOTION_HISTORY_LEN)
        self._speed_history[tid].append(inst)
        return inst

    def _avg_speed(self, tid: int) -> float:
        hist = self._speed_history.get(tid)
        return float(np.mean(hist)) if hist else 0.0

    def _is_closing(self, key: Tuple[int, int], current_dist: float) -> bool:
        if key not in self._dist_history:
            self._dist_history[key] = deque(maxlen=CLOSING_HISTORY_LEN)
        hist = self._dist_history[key]
        hist.append(current_dist)
        if len(hist) < 2:
            return True
        oldest = hist[0]
        if oldest < 1e-6:
            return True
        return (oldest - current_dist) / oldest >= CLOSING_DROP_RATIO

    def _pair_score(self, dist: float, closing: bool, asymmetric: bool) -> float:
        prox  = max(0.0, 1.0 - dist / PROX_DISTANCE_PX)
        sig   = (0.5 if closing else 0.0) + (0.5 if asymmetric else 0.0)
        return round(prox * 0.4 + sig * 0.6, 4)

    def process_frame(
        self,
        frame_idx: int,
        persons:   List[Dict],
    ) -> List[_PersonCandidateEvent]:
        for p in persons:
            self._update_speed(p["track_id"], p["center"])

        events: List[_PersonCandidateEvent] = []
        n = len(persons)

        for i in range(n):
            for j in range(i + 1, n):
                pa, pb = persons[i], persons[j]
                id_a = pa["track_id"]
                id_b = pb["track_id"]
                key  = (min(id_a, id_b), max(id_a, id_b))

                # Cooldown
                if (frame_idx - self._last_event_frame.get(key, -9999)) < EVENT_COOLDOWN:
                    continue

                # Puerta de proximidad
                dist = float(np.linalg.norm(pa["center"] - pb["center"]))
                if dist > PROX_DISTANCE_PX:
                    self._is_closing(key, dist)
                    continue

                closing   = self._is_closing(key, dist)
                sp_a      = self._avg_speed(id_a)
                sp_b      = self._avg_speed(id_b)
                fast      = max(sp_a, sp_b)
                slow      = min(sp_a, sp_b)
                asymmetric = (
                    fast > MOTION_THR_PX
                    and (fast - slow) / (fast + 1e-6) >= ASYMMETRY_RATIO
                )

                if not closing and not asymmetric:
                    continue

                relation = (
                    "contact"    if dist < 60 else
                    "asymmetric" if asymmetric and not closing else
                    "closing"
                )

                events.append(_PersonCandidateEvent(
                    frame_idx=frame_idx,
                    pair_key=key,
                    relation=relation,
                    score=self._pair_score(dist, closing, asymmetric),
                ))
                self._last_event_frame[key] = frame_idx

        return events


# ---------------------------------------------------------------------------
# Verificador ActionCLIP
# ---------------------------------------------------------------------------

class _ActionCLIPVerifier:
    """
    Carga el backbone CLIP + cabeza temporal de ActionCLIP y realiza la
    inferencia sobre un clip de NUM_SEGMENTS frames recortados al par.

    Flujo:
        [T, 3, H, W] → encode_image → [T, D]
                     → unsqueeze(0) → [1, T, D]
                     → fusion_model → [1, D]
                     → cosine sim   → logits [1, 2]
                     → softmax      → pos_prob, neg_prob
    """

    def __init__(
        self,
        checkpoint_path: str,
        arch:            str   = "ViT-B/16",
        sim_header:      str   = "Transf",
        num_segments:    int   = NUM_SEGMENTS,
        pos_threshold:   float = POS_THRESHOLD,
        min_margin:      float = MIN_MARGIN,
        input_size:      int   = 224,
        device:          str   = _DEVICE,
    ):
        self.checkpoint_path = checkpoint_path
        self.arch            = arch
        self.sim_header      = sim_header
        self.num_segments    = num_segments
        self.pos_threshold   = pos_threshold
        self.min_margin      = min_margin
        self.input_size      = input_size
        self.device          = device

        self.model         = None
        self.fusion_model  = None
        self.preprocess    = None
        self.text_features = None   # [2, D]
        self._trt_model    = None   # ActionCLIPModelTRT (si existe)

    def load(self):
        import importlib.util as _ilu

        # The ZeroShot env has openai/CLIP installed as `clip`, which shadows
        # the ActionCLIP fork once it is cached in sys.modules.  Load the fork
        # under a private package name so it never conflicts.
        _pkg = "_arconte_actionclip"
        if _pkg not in sys.modules:
            _clip_dir  = os.path.join(_ACTIONCLIP_REPO, "clip")
            _clip_init = os.path.join(_clip_dir, "__init__.py")
            _spec = _ilu.spec_from_file_location(
                _pkg, _clip_init,
                submodule_search_locations=[_clip_dir],
            )
            actionclip = _ilu.module_from_spec(_spec)
            sys.modules[_pkg] = actionclip
            _spec.loader.exec_module(actionclip)
        else:
            actionclip = sys.modules[_pkg]

        from modules.Visual_Prompt import visual_prompt

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"[RobberyExpert] Checkpoint no encontrado: {self.checkpoint_path}"
            )

        scale = self.input_size * 256 // 224
        self.preprocess = T.Compose([
            T.Resize(scale, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(self.input_size),
            T.ToTensor(),
            T.Normalize(
                mean=(0.48145466, 0.4578275,  0.40821073),
                std =(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        # Backbone
        self.model, _ = actionclip.load(
            self.arch,
            device=self.device,
            jit=False,
            tsm=False,
            T=self.num_segments,
            dropout=0.0,
            emb_dropout=0.0,
        )

        # Cabeza temporal
        self.fusion_model = visual_prompt(
            self.sim_header, self.model.state_dict(), self.num_segments
        ).to(self.device)

        # Pesos del checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)

        def _strip_module(sd):
            if all(k.startswith("module.") for k in sd):
                return {k[len("module."):]: v for k, v in sd.items()}
            return sd

        self.model.load_state_dict(_strip_module(ckpt["model_state_dict"]), strict=True)
        self.fusion_model.load_state_dict(
            _strip_module(ckpt["fusion_model_state_dict"]), strict=True
        )
        self.model.eval()
        self.fusion_model.eval()

        # Prototipos de texto (codificados una vez)
        with torch.no_grad():
            pos_tok = actionclip.tokenize(_POS_PROMPTS).to(self.device)
            neg_tok = actionclip.tokenize(_NEG_PROMPTS).to(self.device)
            pos_f   = self.model.encode_text(pos_tok)
            neg_f   = self.model.encode_text(neg_tok)
            pos_f   = pos_f / pos_f.norm(dim=-1, keepdim=True)
            neg_f   = neg_f / neg_f.norm(dim=-1, keepdim=True)
            pos_p   = pos_f.mean(0, keepdim=True)
            neg_p   = neg_f.mean(0, keepdim=True)
            pos_p   = pos_p / pos_p.norm(dim=-1, keepdim=True)
            neg_p   = neg_p / neg_p.norm(dim=-1, keepdim=True)
            self.text_features = torch.cat([pos_p, neg_p], dim=0)   # [2, D]

        # Detectar engines TRT
        ckpt_dir            = os.path.dirname(self.checkpoint_path) or "."
        visual_engine_path  = os.path.join(ckpt_dir, "ActionCLIP_visual_fp16.engine")
        fusion_engine_path  = os.path.join(ckpt_dir, "ActionCLIP_fusion_fp16.engine")
        if os.path.exists(visual_engine_path) and os.path.exists(fusion_engine_path):
            from core.trt_clip import ActionCLIPModelTRT
            self._trt_model = ActionCLIPModelTRT(
                visual_engine_path, fusion_engine_path,
                num_segments=self.num_segments, feat_dim=512,
            )
            print(
                f"[RobberyExpert] ActionCLIP TensorRT listo. "
                f"T={self.num_segments}  device={self.device}"
            )
        else:
            print(
                f"[RobberyExpert] ActionCLIP PyTorch listo. "
                f"arch={self.arch}  header={self.sim_header}  "
                f"T={self.num_segments}  device={self.device}"
            )

    def _preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        black = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        out   = []
        for fr in frames:
            if fr is None or fr.size == 0:
                fr = black
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            out.append(self.preprocess(Image.fromarray(rgb)))
        return torch.stack(out, dim=0)   # [T, 3, H, W]

    @torch.no_grad()
    def predict(self, frames: List[np.ndarray]) -> Dict:
        """
        Inferencia completa sobre un clip de NUM_SEGMENTS frames recortados.
        Devuelve {"detected", "pos_prob", "neg_prob", "margin"}.
        """
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        if len(frames) < self.num_segments:
            frames = frames + [black.copy()] * (self.num_segments - len(frames))
        else:
            frames = frames[:self.num_segments]

        clip_t = self._preprocess_frames(frames).to(self.device)   # [T, 3, H, W]

        if self._trt_model is not None:
            vid_feat = self._trt_model(clip_t)                     # [1, D]
        else:
            img_feat   = self.model.encode_image(clip_t)           # [T, D]
            img_feat   = img_feat.unsqueeze(0)                     # [1, T, D]
            vid_feat   = self.fusion_model(img_feat)               # [1, D]

        vid_feat = vid_feat / vid_feat.norm(dim=-1, keepdim=True)

        logits   = 100.0 * (vid_feat @ self.text_features.T)       # [1, 2]
        probs    = F.softmax(logits, dim=-1)

        pos_prob = float(probs[0, 0])
        neg_prob = float(probs[0, 1])
        margin   = pos_prob - neg_prob
        detected = (pos_prob >= self.pos_threshold) and (margin >= self.min_margin)

        return {
            "detected": bool(detected),
            "pos_prob": round(pos_prob, 4),
            "neg_prob": round(neg_prob, 4),
            "margin":   round(margin, 4),
        }


# ---------------------------------------------------------------------------
# Helpers de crop
# ---------------------------------------------------------------------------

def _compute_pair_bbox(
    b1: np.ndarray,
    b2: np.ndarray,
    pad_px: int,
    frame_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    H, W = frame_shape
    x1 = max(0, int(min(b1[0], b2[0])) - pad_px)
    y1 = max(0, int(min(b1[1], b2[1])) - pad_px)
    x2 = min(W, int(max(b1[2], b2[2])) + pad_px)
    y2 = min(H, int(max(b1[3], b2[3])) + pad_px)
    return x1, y1, x2, y2


def _crop_frame(
    frame: np.ndarray,
    bbox:  Tuple[int, int, int, int],
    size:  Tuple[int, int] = (640, 480),
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return cv2.resize(crop, size)


def _build_clip(
    frame_buffer:     deque,
    event_global_idx: int,
    pair_bbox:        Tuple[int, int, int, int],
    num_segments:     int = NUM_SEGMENTS,
    pre_frames:       int = PRE_FRAMES,
) -> List[np.ndarray]:
    """
    Ensambla un clip de num_segments frames recortados, centrado en el evento.
    Los frames faltantes se rellenan con negro.
    """
    post_frames = num_segments - pre_frames - 1
    buf_dict    = {e.frame_idx: e.image for e in frame_buffer}
    black       = np.zeros((480, 640, 3), dtype=np.uint8)
    result      = []

    for offset in range(-pre_frames, post_frames + 1):
        idx = event_global_idx + offset
        if idx in buf_dict:
            result.append(_crop_frame(buf_dict[idx], pair_bbox))
        else:
            result.append(black.copy())

    return result


# ---------------------------------------------------------------------------
# RobberyExpert — experto BaseExpert
# ---------------------------------------------------------------------------

class RobberyExpert(BaseExpert):
    """
    Detecta robos con violencia usando:
      1. Heurística pixel-space (_PersonEventDetector): proximidad + cierre/asimetría.
      2. Buffer de frames interno: rolling deque de _FrameEntry.
      3. Inferencia diferida: el clip se arma POST_FRAMES frames después del evento.
      4. ActionCLIP (_ActionCLIPVerifier): clip de 32 frames → softmax pos/neg.
      5. Confirmación: CONSEC_CLIPS_REQUIRED clips positivos consecutivos.
      6. TTL: ROBBERY_TTL frames con etiqueta activa tras confirmación.

    Uso en main.py:
        from experts.robbery_expert import RobberyExpert
        robbery = RobberyExpert(checkpoint_path=".checkpoints/actionclip_robbery.pt")
        experts = [..., robbery]
        # load(), process_heuristics() y is_active siguen el contrato normal.
    """

    def __init__(
        self,
        checkpoint_path: str,
        arch:            str   = "ViT-B/16",
        sim_header:      str   = "Transf",
        pos_threshold:   float = POS_THRESHOLD,
        min_margin:      float = MIN_MARGIN,
        device:          Optional[str] = None,
    ):
        self._checkpoint_path = checkpoint_path
        self._arch            = arch
        self._sim_header      = sim_header
        self._pos_threshold   = pos_threshold
        self._min_margin      = min_margin
        self._device          = device or _DEVICE

        # Componentes internos (inicializados en load())
        self._verifier:        Optional[_ActionCLIPVerifier] = None
        self._event_detector:  _PersonEventDetector = _PersonEventDetector()

        # Buffer de frames
        self._frame_buffer: deque = deque(maxlen=_BUF_CAPACITY)

        # Eventos pendientes de inferencia
        # Cada entrada: {"event": _PersonCandidateEvent, "pair_bbox": tuple, "ready_at": int}
        self._pending: List[Dict] = []

        # Estado de detección
        self._robbery_confirmed: bool  = False
        self._robbery_ttl:       int   = 0
        self._consec_pos:        int   = 0
        self._last_crop:         Optional[np.ndarray] = None

        # Worker thread
        self._q: queue.Queue = queue.Queue(maxsize=2)

    # ------------------------------------------------------------------
    # BaseExpert — propiedades obligatorias
    # ------------------------------------------------------------------

    @property
    def label(self) -> str:
        return "ROBO_V"

    @property
    def is_active(self) -> bool:
        return self._robbery_confirmed

    # ------------------------------------------------------------------
    # BaseExpert — load
    # ------------------------------------------------------------------

    def load(self, model: Any, preprocess: Any) -> None:
        """
        Carga ActionCLIP desde el checkpoint y lanza el worker thread.
        Los parámetros model/preprocess (CLIP compartido) no se usan:
        ActionCLIP gestiona su propio backbone ViT.
        """
        self._verifier = _ActionCLIPVerifier(
            checkpoint_path=self._checkpoint_path,
            arch=self._arch,
            sim_header=self._sim_header,
            pos_threshold=self._pos_threshold,
            min_margin=self._min_margin,
            device=self._device,
        )
        self._verifier.load()
        threading.Thread(target=self._worker_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # BaseExpert — process_heuristics
    # ------------------------------------------------------------------

    def process_heuristics(
        self,
        frame:        np.ndarray,
        tracker_data: Dict[str, Any],
    ) -> None:
        """
        Llamado cada frame por el orquestador.

        Pasos:
          1. Añadir frame al buffer interno.
          2. Convertir tracker_data → lista de dicts de persona.
          3. Ejecutar _PersonEventDetector.
          4. Programar inferencia POST_FRAMES frames después de cada evento.
          5. Disparar inferencia para eventos que ya tienen el clip completo.
          6. Decrementar TTL.
        """
        frame_idx = tracker_data["frame_idx"]
        H, W      = frame.shape[:2]

        # 1. Buffer de frames
        self._frame_buffer.append(_FrameEntry(image=frame.copy(), frame_idx=frame_idx))

        # 2. Personas como lista de dicts
        persons = self._persons_from_tracker(tracker_data)

        # 3. Detector de eventos
        events = self._event_detector.process_frame(frame_idx, persons)

        for ev in events:
            id_a, id_b = ev.pair_key
            bbox_map   = {p["track_id"]: p["bbox_xyxy"] for p in persons}
            b1 = bbox_map.get(id_a)
            b2 = bbox_map.get(id_b)
            if b1 is None or b2 is None:
                continue

            pair_bbox        = _compute_pair_bbox(b1, b2, CROP_PAD_PX, (H, W))
            self._last_crop  = _crop_frame(frame, pair_bbox)

            self._pending.append({
                "event":     ev,
                "pair_bbox": pair_bbox,
                "ready_at":  frame_idx + POST_FRAMES + 1,
            })

        # 4. Despachar eventos listos al worker
        still_pending = []
        for pe in self._pending:
            if frame_idx >= pe["ready_at"]:
                clip_frames = _build_clip(
                    self._frame_buffer,
                    pe["event"].frame_idx,
                    pe["pair_bbox"],
                )
                try:
                    self._q.put_nowait(clip_frames)
                except queue.Full:
                    pass   # worker ocupado — descartamos este clip
            else:
                still_pending.append(pe)
        self._pending = still_pending

        # 5. TTL
        if self._robbery_confirmed:
            self._robbery_ttl -= 1
            if self._robbery_ttl <= 0:
                self._robbery_confirmed = False
                self._robbery_ttl       = 0
                self._consec_pos        = 0

    # ------------------------------------------------------------------
    # BaseExpert — predict
    # ------------------------------------------------------------------

    def predict(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Inferencia ActionCLIP sobre un clip de NUM_SEGMENTS frames.
        Llamado exclusivamente por el worker thread interno.
        Actualiza el estado de confirmación como efecto secundario.
        """
        result = self._verifier.predict(frames)

        if result["detected"]:
            self._consec_pos += 1
            if self._consec_pos >= CONSEC_CLIPS_REQUIRED:
                self._robbery_confirmed = True
                self._robbery_ttl       = ROBBERY_TTL
                print(
                    f"[RobberyExpert] *** ROBO CONFIRMADO ***  "
                    f"pos={result['pos_prob']:.4f}  margin={result['margin']:+.4f}"
                )
        else:
            self._consec_pos = 0

        return {
            "detected": result["detected"],
            "score":    result["pos_prob"],
            "margin":   result["margin"],
        }

    # ------------------------------------------------------------------
    # get_display_data
    # ------------------------------------------------------------------

    def get_display_data(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"color": (0, 0, 180)}
        if self._last_crop is not None:
            data["last_crop"] = self._last_crop
        return data

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            clip_frames = self._q.get()
            try:
                self.predict(clip_frames)
            except Exception as exc:
                print(f"[RobberyExpert] Error en inferencia: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _persons_from_tracker(tracker_data: Dict[str, Any]) -> List[Dict]:
        """
        Convierte arrays del tracker al formato de dicts que espera
        _PersonEventDetector: {"track_id", "bbox_xyxy", "center"}.
        """
        persons = []
        for xyxy, tid in zip(
            tracker_data["persons_xyxy"],
            tracker_data["persons_ids"],
        ):
            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0
            persons.append({
                "track_id":  int(tid),
                "bbox_xyxy": xyxy,
                "center":    np.array([cx, cy], dtype=np.float32),
            })
        return persons
