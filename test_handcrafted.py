#!/home/pc/miniconda3/envs/celestial_v2/bin/python
"""
test_handcrafted.py
===================
Evaluación de los expertos Arconte V2.0.0 sobre el test set de UCF-Crime.

Métricas computadas
-------------------
  - evaluate_frame_auc   : AUC-ROC global y por video (nivel de frame)
  - evaluate_video_accuracy: Matriz de confusión + Classification Report (nivel de video)

Salidas
-------
  results/handcrafted_metrics.txt           — reporte completo
  results/plots/<VideoStem>_score_curve.png — curva de score por video

Lógica de ejecución
--------------------
  1. Parsea Temporal_Anomaly_Annotation.txt y filtra categorías soportadas.
  2. Carga CLIP ViT-L/14 + todos los expertos + ArconteTracker (una sola vez).
  3. Por video: resetea estado, procesa ventanas de CLIP_LEN frames con
     tracker + process_heuristics + predict() síncrono.
  4. El score normalizado de cada ventana se asigna a todos sus frames.
  5. Se calculan métricas y se generan gráficas con matplotlib.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Asegura que el clip bundled en ActionCLIP/ sea importable
_ROOT_EARLY = Path(__file__).parent
sys.path.insert(0, str(_ROOT_EARLY / "ActionCLIP"))
sys.path.insert(0, str(_ROOT_EARLY))

import cv2
import clip  # type: ignore
import matplotlib
matplotlib.use("Agg")  # sin display — debe ir antes de importar pyplot
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (  # type: ignore
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

# ── Raíz del proyecto ─────────────────────────────────────────────────────────
_ROOT = _ROOT_EARLY  # ya insertado en sys.path arriba

from core.base_expert import BaseExpert
from core.tracker import ArconteTracker
from experts_V2_0_0.carparts_expert import CarPartsExpert
from experts_V2_0_0.crash_expert import CrashExpert
from experts_V2_0_0.fight_expert import FightExpert
from experts_V2_0_0.fire_expert import make_fire_smoke_experts
from experts_V2_0_0.robbery_expert import RobberyExpert

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Rutas ─────────────────────────────────────────────────────────────────────
ANNOTATION_FILE = _ROOT / "resources" / "Temporal_Anomaly_Annotation.txt"
VIDEO_ROOT      = Path("/media/pc/backup1/BaseDeDatos/UCF-Crime/Videos")
CLIP_CKPT       = _ROOT / ".checkpoints" / "ViT-L-14.pt"
ROBBERY_CKPT    = _ROOT / ".checkpoints" / "ViT-B-16-32-f.pt"
RESULTS_DIR     = _ROOT / "results"
PLOTS_DIR       = RESULTS_DIR / "plots"

# ── Parámetros de clip (igual que data_processor.py) ─────────────────────────
CLIP_LEN    = 16    # frames por ventana
STRIDE      = 2     # stride temporal entre frames del clip
CLIP_STEP   = 16    # paso entre inicios de ventanas consecutivas
OUTPUT_SIZE = 224   # resolución de entrada al experto

# ── Mapeo de categorías (igual que data_processor.py) ────────────────────────
CATEGORY_MAP: Dict[str, str] = {
    "Fighting":      "fight",
    "Assault":       "fight",
    "RoadAccidents": "crash",
    "Arson":         "fire",
    "Explosion":     "fire",
    "Burglary":      "robbery",
    "Robbery":       "robbery",
    "Stealing":      "carparts",
    "Normal":        "normal",
}

# ── Umbrales de alerta por experto (para normalización de score) ──────────────
# Un score normalizado ≥ 1.0 equivale a cruzar el umbral del experto.
EXPERT_THRESHOLDS: Dict[str, float] = {
    "fight":    3.8,    # F_SET_SCORE_TH
    "crash":    0.90,   # C_MIN_SET_SCORE
    "fire":     3.5,    # FIRE_CLIP_SET_SCORE_TH
    "carparts": 0.50,   # ALERT_THRESHOLD
    "robbery":  0.50,   # POS_THRESHOLD
    "normal":   1.0,    # sin umbral — usamos el máximo entre expertos
}


# ═══════════════════════════════════════════════════════════════════════════════
# Estructura de anotación
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VideoEntry:
    name:        str
    category:    str
    expert_type: str                        # "fight" | "crash" | … | "normal"
    ranges:      List[Tuple[int, int]]      # [(start, end), …], vacío si Normal
    gt_label:    int = field(init=False)    # 1 = anomalía, 0 = normal

    def __post_init__(self) -> None:
        self.gt_label = 1 if self.ranges else 0


def parse_annotations(path: Path) -> List[VideoEntry]:
    """
    Parsea Temporal_Anomaly_Annotation.txt.

    Formato: VideoName  Category  Start1  End1  Start2  End2
    Ejemplo: Arson011_x264.mp4  Arson  150  420  680  1266
    Normal : Normal_Videos_003_x264.mp4  Normal  -1  -1  -1  -1
    """
    entries: List[VideoEntry] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 6:
                continue
            name, category = parts[0], parts[1]
            s1, e1, s2, e2 = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])

            expert_type = CATEGORY_MAP.get(category)
            if expert_type is None:
                log.debug("Categoría no soportada, skip: %s (%s)", name, category)
                continue

            ranges: List[Tuple[int, int]] = []
            if s1 != -1:
                ranges.append((s1, e1))
            if s2 != -1:
                ranges.append((s2, e2))

            entries.append(VideoEntry(
                name=name,
                category=category,
                expert_type=expert_type,
                ranges=ranges,
            ))

    log.info("Anotaciones parseadas: %d videos (%d categorías soportadas)",
             len(entries), len({e.category for e in entries}))
    return entries


def find_video(name: str, category: str, root: Path) -> Optional[Path]:
    """
    Localiza el archivo de video en ROOT/{Category}/{name}.
    Estructura real del dataset: Videos/{Category}/  (Normal → "Normal").
    """
    stem = Path(name).stem

    # Intento directo con el directorio de categoría y extensiones alternativas
    for ext in ("", ".mp4", ".avi", ".mkv"):
        p = root / category / (stem + ext if ext else name)
        if p.exists():
            return p

    # Búsqueda recursiva como respaldo (cubre variaciones de estructura)
    for ext in (".mp4", ".avi", ".mkv"):
        for p in root.rglob(stem + ext):
            return p

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Supresión de stdout (expertos imprimen internamente)
# ═══════════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as dn:
        old, sys.stdout = sys.stdout, dn
        try:
            yield
        finally:
            sys.stdout = old


# ═══════════════════════════════════════════════════════════════════════════════
# Carga de modelos
# ═══════════════════════════════════════════════════════════════════════════════

def load_models(
    device: str,
) -> Tuple[Any, Any, Dict[str, BaseExpert], ArconteTracker]:
    """
    Carga CLIP, todos los expertos y el tracker.
    Solo se llama una vez antes del bucle principal.
    """
    log.info("Cargando CLIP ViT-L/14 en %s …", device)
    with _quiet():
        clip_model, clip_preprocess = clip.load(
            str(CLIP_CKPT), device=device, jit=False
        )
    clip_model.eval()

    log.info("Inicializando expertos …")
    fire_expert, _ = make_fire_smoke_experts()
    experts: Dict[str, BaseExpert] = {
        "fight":    FightExpert(),
        "crash":    CrashExpert(),
        "fire":     fire_expert,
        "carparts": CarPartsExpert(),
        "robbery":  RobberyExpert(checkpoint_path=str(ROBBERY_CKPT)),
    }
    with _quiet():
        for exp in experts.values():
            exp.load(clip_model, clip_preprocess)

    log.info("Cargando ArconteTracker (YOLO) …")
    with _quiet():
        tracker = ArconteTracker()

    log.info("Todos los modelos listos.")
    return clip_model, clip_preprocess, experts, tracker


# ═══════════════════════════════════════════════════════════════════════════════
# Reset de estado entre videos
# ═══════════════════════════════════════════════════════════════════════════════

def reset_expert(expert: BaseExpert) -> None:
    """
    Reinicia el estado temporal del experto entre videos.
    Usa hasattr para ser compatible con cualquier versión de experto.
    """
    # Atributos numéricos → 0.0
    for attr in (
        "_smoothed_score", "_last_score", "_consec_pos",
        "_sticky_counter",
        # Crash
        "crash_counter", "crash_ttl_counter", "crash_consec_pos",
        "crash_active_frames", "candidate_sent_count",
        # Fire / Smoke
        "fire_clip_consec_pos", "fire_clip_ttl",
        "smoke_clip_consec_pos", "smoke_clip_ttl",
        # CarParts
        "_last_ascore", "_last_score_kf", "_last_score_pc", "_last_score_tc",
    ):
        if hasattr(expert, attr):
            setattr(expert, attr, 0.0)

    # Atributos booleanos → False
    for attr in (
        "_is_detected", "_target_locked", "_crash_processing", "_processing",
        "crash_locked", "crash_is_detected", "crash_ever_detected",
        "fire_clip_confirmed", "smoke_clip_confirmed",
        "_robbery_confirmed",
    ):
        if hasattr(expert, attr):
            setattr(expert, attr, False)

    # Atributos None
    for attr in (
        "_last_crop", "_last_valid_bbox", "_locked_ids",
        "_last_kf_bgr", "_last_attn_ovl", "_last_hot_bbox",
        "crash_last_crop", "crash_interaction_type",
        "active_car_pair", "active_pileup",
    ):
        if hasattr(expert, attr):
            setattr(expert, attr, None)

    # Atributos dict → {}
    for attr in (
        "_prev_boxes", "crash_car_prev_center", "_motion_history",
        "_pair_dist_history",
    ):
        if hasattr(expert, attr):
            setattr(expert, attr, {})

    # Atributos lista → []
    for attr in (
        "_buffer", "_score_history",
        "crash_buffer", "crashed_stationary_cars", "_prev_car_boxes",
    ):
        if hasattr(expert, attr):
            setattr(expert, attr, [])

    # Tuplas especiales
    if hasattr(expert, "_last_ids"):
        setattr(expert, "_last_ids", (-1, -1))

    # Vaciar colas de worker sin bloquear
    for attr in ("_q",):
        q = getattr(expert, attr, None)
        if q is not None:
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break


# ═══════════════════════════════════════════════════════════════════════════════
# Procesamiento de video → scores por frame
# ═══════════════════════════════════════════════════════════════════════════════

def score_video(
    video_path:  Path,
    expert_type: str,
    experts:     Dict[str, BaseExpert],
    tracker:     ArconteTracker,
) -> Tuple[np.ndarray, bool]:
    """
    Procesa el video y devuelve (frame_scores_norm, video_detected).

    frame_scores_norm : float32 [total_frames]  score normalizado por umbral.
    video_detected    : True si algún clip superó el umbral del experto.

    Metodología (igual a data_processor.py):
      - Por cada ventana de CLIP_LEN*STRIDE frames (paso CLIP_STEP):
          1. tracker.process() + expert.process_heuristics() sobre frames muestreados.
          2. expert.predict() síncrono con los frames redimensionados a 224×224.
          3. score normalizado = score_raw / umbral_experto.
          4. El score se asigna a todos los frames de la ventana.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("No se puede abrir: %s", video_path)
        return np.zeros(0, dtype=np.float32), False

    raw_frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_frames.append(frame)
    cap.release()

    total = len(raw_frames)
    frame_scores = np.zeros(total, dtype=np.float32)
    video_detected = False

    min_frames = CLIP_LEN * STRIDE
    if total < min_frames:
        log.debug("Video demasiado corto (%d frames): %s", total, video_path.name)
        return frame_scores, False

    # Expertos activos: uno o todos (normal)
    threshold = EXPERT_THRESHOLDS.get(expert_type, 1.0)
    active = experts if expert_type == "normal" else {expert_type: experts[expert_type]}

    # Reset tracker frame index (state de ByteTrack persiste entre videos,
    # pero los IDs son reasignados automáticamente al perder los tracks)
    tracker._frame_idx = 0

    for start in range(0, total - min_frames + 1, CLIP_STEP):
        src_idx = [start + i * STRIDE for i in range(CLIP_LEN)]

        # ── Tracker + heurísticas ────────────────────────────────────────────
        last_td: Dict[str, Any] = {
            "persons_xyxy":      np.empty((0, 4), dtype=np.float32),
            "persons_ids":       np.empty((0,),   dtype=np.int32),
            "vehicles_xyxy":     np.empty((0, 4), dtype=np.float32),
            "vehicles_ids":      np.empty((0,),   dtype=np.int32),
            "vehicles_confs":    np.empty((0,),   dtype=np.float32),
            "vehicles_all_xyxy": np.empty((0, 4), dtype=np.float32),
            "vehicles_all_ids":  np.empty((0,),   dtype=np.int32),
            "frame_idx":         0,
        }
        for idx in src_idx:
            try:
                with _quiet():
                    td = tracker.process(raw_frames[idx])
                last_td = td
            except Exception:
                pass
            for exp in active.values():
                try:
                    with _quiet():
                        exp.process_heuristics(raw_frames[idx], last_td)
                except Exception:
                    pass

        # ── Resize → predict() síncrono ─────────────────────────────────────
        resized = [
            cv2.resize(raw_frames[i], (OUTPUT_SIZE, OUTPUT_SIZE))
            for i in src_idx
        ]

        end = min(start + CLIP_STEP, total)

        if expert_type == "normal":
            # Normal: puntaje = max(score_normalizado) de todos los expertos
            max_norm = 0.0
            for etype, exp in active.items():
                try:
                    with _quiet():
                        r = exp.predict(resized)
                    raw  = float(r.get("score", 0.0))
                    norm = raw / EXPERT_THRESHOLDS.get(etype, 1.0)
                    max_norm = max(max_norm, norm)
                    if r.get("detected", False):
                        video_detected = True
                except Exception:
                    pass
            frame_scores[start:end] = max_norm

        else:
            exp = active[expert_type]
            try:
                with _quiet():
                    r = exp.predict(resized)
                raw_score  = float(r.get("score", 0.0))
                norm_score = raw_score / threshold
                detected   = bool(r.get("detected", False))
            except Exception:
                norm_score = 0.0
                detected   = False

            frame_scores[start:end] = norm_score
            if detected:
                video_detected = True

    return frame_scores, video_detected


# ═══════════════════════════════════════════════════════════════════════════════
# GT vector
# ═══════════════════════════════════════════════════════════════════════════════

def build_gt_vector(
    total_frames: int,
    ranges: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Construye el vector Ground Truth binario de longitud `total_frames`.

    Las anotaciones UCF-Crime usan índices 1-basados inclusive,
    por lo que se hace la corrección a 0-basados:  start-1 … end (exclusive).
    Si ranges está vacío, el vector es todo ceros (video Normal).
    """
    gt = np.zeros(total_frames, dtype=np.int32)
    for s, e in ranges:
        # Anotación 1-based → 0-based: frames [s-1, e-1] inclusive
        s0 = max(0, s - 1)
        e0 = min(total_frames, e)   # slicing exclusive
        gt[s0:e0] = 1
    return gt


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluación a nivel de Frame (AUC-ROC)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_frame_auc(
    results: List[Dict[str, Any]],
    plots_dir: Path,
) -> Dict[str, Any]:
    """
    Computa AUC-ROC global y por video.

    results: lista de dicts con claves:
        "name", "expert_type", "scores" (np.ndarray), "gt" (np.ndarray),
        "ranges" (list of tuples), "total_frames" (int)

    Devuelve dict con:
        "global_auc"   : float
        "per_video_auc": dict[str → float]
        "fpr", "tpr"   : para la curva global
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_scores: List[float] = []
    all_gt:     List[int]   = []
    per_video_auc: Dict[str, float] = {}

    for rec in results:
        scores = rec["scores"]
        gt     = rec["gt"]
        name   = rec["name"]
        ranges = rec["ranges"]

        # Acumular para AUC global
        all_scores.extend(scores.tolist())
        all_gt.extend(gt.tolist())

        # AUC por video (solo si hay frames de ambas clases)
        n_pos = int(gt.sum())
        n_neg = int((gt == 0).sum())
        if n_pos > 0 and n_neg > 0:
            try:
                video_auc = float(roc_auc_score(gt, scores))
            except Exception:
                video_auc = float("nan")
            per_video_auc[name] = video_auc
        else:
            per_video_auc[name] = float("nan")

        # ── Gráfica por video ────────────────────────────────────────────────
        _plot_video_scores(
            name=name,
            scores=scores,
            gt=gt,
            ranges=ranges,
            expert_type=rec["expert_type"],
            out_path=plots_dir / (Path(name).stem + "_score_curve.png"),
        )

    # AUC global
    all_scores_arr = np.array(all_scores, dtype=np.float32)
    all_gt_arr     = np.array(all_gt,     dtype=np.int32)

    n_pos_total = int(all_gt_arr.sum())
    n_neg_total = int((all_gt_arr == 0).sum())

    if n_pos_total == 0 or n_neg_total == 0:
        global_auc = float("nan")
        fpr = tpr = np.array([])
    else:
        global_auc = float(roc_auc_score(all_gt_arr, all_scores_arr))
        fpr, tpr, _ = roc_curve(all_gt_arr, all_scores_arr)

    # Curva ROC global
    if fpr.size > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {global_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Curva ROC Global — Arconte V2.0.0 / UCF-Crime")
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(plots_dir / "global_roc_curve.png"), dpi=120)
        plt.close(fig)

    return {
        "global_auc":    global_auc,
        "per_video_auc": per_video_auc,
        "fpr":           fpr,
        "tpr":           tpr,
    }


def _plot_video_scores(
    name:        str,
    scores:      np.ndarray,
    gt:          np.ndarray,
    ranges:      List[Tuple[int, int]],
    expert_type: str,
    out_path:    Path,
) -> None:
    """Genera la gráfica de score vs frame con sombreado de anomalía real."""
    fig, ax = plt.subplots(figsize=(15, 4))
    frames = np.arange(len(scores))

    ax.plot(frames, scores, lw=1.2, color="#1f77b4", label="Score normalizado")
    ax.axhline(1.0, color="red", lw=1, ls="--", label="Umbral del experto")

    # Sombreado rojo en rangos de anomalía
    for s, e in ranges:
        s0 = max(0, s - 1)
        e0 = min(len(scores), e)
        ax.axvspan(s0, e0, alpha=0.25, color="red", label="Anomalía GT")

    # Evitar etiquetas duplicadas
    handles, labels = ax.get_legend_handles_labels()
    seen: Dict[str, bool] = {}
    unique = [(h, l) for h, l in zip(handles, labels) if not seen.setdefault(l, False)]
    ax.legend(*zip(*unique) if unique else ([], []), fontsize=8)

    ax.set_xlim(0, len(scores))
    ax.set_xlabel("Frame")
    ax.set_ylabel("Score / Umbral")
    ax.set_title(f"{name}  [experto: {expert_type}]", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluación a nivel de Video (clasificación binaria)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_video_accuracy(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Clasifica cada video como Normal (0) o Anomalía (1) y calcula:
      - Matriz de confusión
      - Classification Report (Precision, Recall, F1)

    Predicción: un video es positivo si video_detected == True
    (es decir, al menos un clip superó el umbral del experto).
    """
    y_true = [r["gt_label"]       for r in results]
    y_pred = [int(r["vid_detect"]) for r in results]

    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=["Normal", "Anomalía"],
        zero_division=0,
    )
    return {"confusion_matrix": cm, "report": report}


# ═══════════════════════════════════════════════════════════════════════════════
# Reporte de texto
# ═══════════════════════════════════════════════════════════════════════════════

def save_report(
    frame_metrics:  Dict[str, Any],
    video_metrics:  Dict[str, Any],
    results:        List[Dict[str, Any]],
    out_path:       Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    lines += [
        "=" * 72,
        "ARCONTE V2.0.0 — Evaluación sobre UCF-Crime Test Set",
        "=" * 72,
        "",
        "── MÉTRICAS A NIVEL DE FRAME ────────────────────────────────────",
        f"  AUC-ROC Global : {frame_metrics['global_auc']:.4f}",
        "",
        "  AUC por video (solo videos con frames de ambas clases):",
    ]
    per_auc = frame_metrics["per_video_auc"]
    valid   = {k: v for k, v in per_auc.items() if not np.isnan(v)}
    for name, a in sorted(valid.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"    {name:<50s}  AUC={a:.4f}")

    if valid:
        lines.append(f"\n  Media AUC por video : {np.mean(list(valid.values())):.4f}")
    lines.append("")

    lines += [
        "── MÉTRICAS A NIVEL DE VIDEO ────────────────────────────────────",
        "  Matriz de confusión (filas=GT, cols=Pred):",
        "               Pred Normal  Pred Anomalía",
    ]
    cm = video_metrics["confusion_matrix"]
    if cm.size == 4:
        lines.append(f"  GT Normal     {cm[0,0]:>10d}  {cm[0,1]:>13d}")
        lines.append(f"  GT Anomalía   {cm[1,0]:>10d}  {cm[1,1]:>13d}")
    lines += ["", "  Classification Report:", ""]
    lines += ["    " + l for l in video_metrics["report"].splitlines()]
    lines += [
        "",
        "── DETALLE POR VIDEO ────────────────────────────────────────────",
    ]
    for r in results:
        gt_lbl  = "ANOMALÍA" if r["gt_label"] else "Normal  "
        pd_lbl  = "ANOMALÍA" if r["vid_detect"] else "Normal  "
        correct = "OK " if r["gt_label"] == int(r["vid_detect"]) else "ERR"
        auc_s   = f"{per_auc.get(r['name'], float('nan')):.4f}"
        lines.append(
            f"  {correct} GT={gt_lbl} Pred={pd_lbl} AUC={auc_s}  {r['name']}"
        )
    lines += ["", "=" * 72]

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("Reporte guardado en %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Dispositivo ──────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        log.warning("CUDA no disponible — usando CPU (evaluación muy lenta).")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        log.info("GPU: %s", gpu_name)

    # ── Parsear anotaciones ───────────────────────────────────────────────────
    entries = parse_annotations(ANNOTATION_FILE)
    log.info("Videos a evaluar: %d", len(entries))

    # ── Cargar modelos ────────────────────────────────────────────────────────
    clip_model, clip_preprocess, experts, tracker = load_models(device)

    # ── Bucle principal ───────────────────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for entry in tqdm(entries, desc="Evaluando videos", unit="vid"):
        video_path = find_video(entry.name, entry.category, VIDEO_ROOT)
        if video_path is None:
            log.warning("Video no encontrado: %s", entry.name)
            continue

        # Resetear estado de expertos relevantes
        if entry.expert_type == "normal":
            for exp in experts.values():
                reset_expert(exp)
        else:
            reset_expert(experts[entry.expert_type])

        # Procesar video
        frame_scores, video_detected = score_video(
            video_path,
            entry.expert_type,
            experts,
            tracker,
        )

        total_frames = len(frame_scores)
        if total_frames == 0:
            log.warning("Sin frames: %s", entry.name)
            continue

        gt_vector = build_gt_vector(total_frames, entry.ranges)

        results.append({
            "name":         entry.name,
            "category":     entry.category,
            "expert_type":  entry.expert_type,
            "scores":       frame_scores,
            "gt":           gt_vector,
            "gt_label":     entry.gt_label,
            "vid_detect":   video_detected,
            "ranges":       entry.ranges,
            "total_frames": total_frames,
        })

        # Liberar caché de GPU entre videos para evitar fugas de memoria
        if device == "cuda":
            torch.cuda.empty_cache()

    if not results:
        log.error("Sin resultados. Verificar rutas y anotaciones.")
        return

    log.info("Procesados %d videos. Calculando métricas …", len(results))

    # ── Métricas a nivel de frame ─────────────────────────────────────────────
    frame_metrics = evaluate_frame_auc(results, PLOTS_DIR)
    log.info("AUC-ROC Global (frame-level): %.4f", frame_metrics["global_auc"])

    # ── Métricas a nivel de video ─────────────────────────────────────────────
    video_metrics = evaluate_video_accuracy(results)
    log.info("Classification Report (video-level):\n%s", video_metrics["report"])

    # ── Guardar reporte ───────────────────────────────────────────────────────
    save_report(
        frame_metrics,
        video_metrics,
        results,
        RESULTS_DIR / "handcrafted_metrics.txt",
    )

    log.info("Gráficas guardadas en %s", PLOTS_DIR)
    log.info("Evaluación completada.")


if __name__ == "__main__":
    main()
