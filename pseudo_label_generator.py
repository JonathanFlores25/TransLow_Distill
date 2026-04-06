    #!/usr/bin/env python3
"""
pseudo_label_generator.py
=========================
Genera pseudo-labels para fine-tuning de VideoMAE en anomaly detection.

Pipeline
--------
  Step 1 — Expert Scoring (pesado, solo corre una vez)
      Corre expertos Arconte en cada clip del dataset.
      Guarda: expert_scores.npy  →  {video_path: [{clip_name, is_active, score, start}]}
      Usa --skip-scoring para reusar si ya existe.

  Step 2 — Distribución normal (VideoMAE features)
      Carga features de clips normales → ajusta Gaussiana sobre norma L2.
      Compatible con metodología C2FPL.

  Step 3 — GT Generation
      Por cada video anómalo:
        · p-valores por clip (norma L2 vs distribución normal)
        · sliding window 20%  →  segmento más anómalo
        · intersección con expert.is_active + score:
            en ventana  AND experto activo   →  label = score  (soft, 0–1)
            en ventana  AND experto inactivo →  descartar      (incierto)
            fuera ventana OR experto inactivo→  label = 0.0
      Clips normales → siempre 0.0

  Step 4 — Visual clips MP4
      Para clips con label > 0 guarda MP4 original + overlay de categoría/score/p-valor.
      Permite verificar visualmente si las anomalías son reales.

Outputs  (data/pseudo_labels/)
-------
  expert_scores.npy    — dict caché de expert scoring
  gt_binary.npy        — float32 (N,)   labels 0.0–1.0   [C2FPL compatible]
  gt_multiclass.npy    — int32   (N,)   clase 0–5         [C2FPL compatible]
  nalist.npy           — int32   (M,2)  (from,to) por video
  clip_names.npy       — str     (N,)   nombre de cada clip en orden
  visual_clips/{cat}/  — MP4s de anomalías confirmadas

Uso
---
  python pseudo_label_generator.py
  python pseudo_label_generator.py --skip-scoring --limit 20
  python pseudo_label_generator.py --p-threshold 0.03 --window 0.25
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import logging.handlers
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.stats import norm as scipy_norm
from sklearn.mixture import GaussianMixture

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import clip as clip_lib  # noqa: E402

from core.base_expert import BaseExpert
from core.tracker import ArconteTracker
from experts_V2_0_0.carparts_expert import CarPartsExpert
from experts_V2_0_0.crash_expert import CrashExpert
from experts_V2_0_0.fight_expert import FightExpert
from experts_V2_0_0.fire_expert import make_fire_smoke_experts
from experts_V2_0_0.robbery_expert import RobberyExpert

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_TXT   = _ROOT / "resources" / "Anomaly_Train.txt"
VIDEO_ROOT    = Path("/media/pc/backup1/BaseDeDatos/UCF-Crime/Videos")
VMAE_FEATURES = _ROOT / "data" / "vmae_features"
OUTPUT_PATH   = _ROOT / "data" / "pseudo_labels"
CLIP_CKPT     = _ROOT / ".checkpoints" / "ViT-L-14.pt"
ROBBERY_CKPT  = _ROOT / ".checkpoints" / "ViT-B-16-32-f.pt"

# ── Clip params (mismos que data_processor.py y FE_VideoMAE.py) ───────────────
CLIP_LEN    = 16
STRIDE      = 2
CLIP_STEP   = 16
OUTPUT_SIZE = 224

# ── Defaults ─────────────────────────────────────────────────────────────────
NUM_WORKERS    = 3
MAX_VRAM_GB    = 11.5
WINDOW_RATIO   = 0.2     # 20% del video = ventana C2FPL
P_THRESHOLD    = 0.05    # p-valor < este → estadísticamente anómalo
GMM_COMPONENTS = 5

# ── Mapeos ───────────────────────────────────────────────────────────────────
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

CLASS_ID: Dict[str, int] = {
    "normal":   0,
    "fight":    1,
    "crash":    2,
    "fire":     3,
    "robbery":  4,
    "carparts": 5,
}

CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "fight":    (0,   0,   255),
    "crash":    (0,  165,  255),
    "fire":     (0,   60,  255),
    "robbery":  (0,   0,   180),
    "carparts": (128,  0,  128),
    "normal":   (0,  200,    0),
}


# ────────────────────────────────────────────────────────────────────────────
# Utilidades
# ────────────────────────────────────────────────────────────────────────────

CKPT_DIR = OUTPUT_PATH / "ckpt_scoring"


def _ckpt_path(video_path: Path) -> Path:
    """Ruta del checkpoint por video (un .npy con los clip results)."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    return CKPT_DIR / f"{video_path.stem}.npy"


@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


def resize224(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (OUTPUT_SIZE, OUTPUT_SIZE))


# ────────────────────────────────────────────────────────────────────────────
# Step 1 — Expert Scoring
# ────────────────────────────────────────────────────────────────────────────

def _score_video(
    video_path:  Path,
    expert_type: str,
    experts:     Dict[str, BaseExpert],
    tracker:     ArconteTracker,
) -> List[Dict[str, Any]]:
    """
    Corre expertos en cada clip del video.
    Retorna lista de dicts {clip_name, start, is_active, score}.
    NO guarda frames ni attn_mask — solo escalares.
    """
    if not video_path.exists():
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
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
        return []

    stem   = video_path.stem
    active = experts if expert_type == "normal" else {expert_type: experts[expert_type]}
    results: List[Dict[str, Any]] = []

    for start in range(0, len(raw_frames) - min_frames + 1, CLIP_STEP):
        src_idx = [start + i * STRIDE for i in range(CLIP_LEN)]

        # Tracker + heurísticas (actualiza estado del experto frame a frame)
        last_td: Dict[str, Any] = {}
        clip_active = False
        for idx in src_idx:
            frame = raw_frames[idx]
            try:
                with _quiet():
                    td = tracker.process(frame)
                last_td = td
            except Exception:
                td = last_td

            for expert in active.values():
                try:
                    with _quiet():
                        expert.process_heuristics(frame, td)
                except Exception:
                    pass
                if expert.is_active:
                    clip_active = True

        # Score de predicción sobre el clip completo
        resized = [resize224(raw_frames[i]) for i in src_idx]
        score   = 0.0

        if expert_type == "normal":
            for expert in active.values():
                try:
                    with _quiet():
                        r = expert.predict(resized)
                    score = max(score, float(r.get("score", 0.0)))
                except Exception:
                    pass
        else:
            expert = active[expert_type]
            try:
                with _quiet():
                    r = expert.predict(resized)
                score = float(r.get("score", 0.0))
            except Exception:
                score = 0.0

        results.append({
            "clip_name": f"{stem}_s{start:06d}",
            "start":     start,
            "is_active": clip_active,
            "score":     score,
        })

        if clip_active:
            score = max(0.0, score)   # clamp inferior: scores negativos no tienen sentido
            results[-1]["score"] = score
            logging.info("  DETECCION  %s_s%06d  score=%.3f  [%s]",
                         stem, start, score, expert_type.upper())

    n_detected = sum(1 for r in results if r["is_active"])
    logging.info("  RESUMEN %s: %d/%d clips con activacion experta",
                 stem, n_detected, len(results))

    return results


def _worker_main(
    worker_id:     int,
    video_queue:   "mp.Queue[Any]",
    results_queue: "mp.Queue[Any]",
    log_queue:     "mp.Queue[Any]",
    max_vram_gb:   float,
    num_workers:   int,
) -> None:
    # Logging via queue
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(logging.handlers.QueueHandler(log_queue))

    # VRAM budget
    if torch.cuda.is_available():
        total  = torch.cuda.get_device_properties(0).total_memory
        budget = (max_vram_gb / num_workers) * (1024 ** 3)
        torch.cuda.set_per_process_memory_fraction(min(budget / total, 0.95))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with _quiet():
        clip_model, clip_preprocess = clip_lib.load(str(CLIP_CKPT), device=device, jit=False)
    clip_model.eval()

    fire_expert, _ = make_fire_smoke_experts()
    experts: Dict[str, BaseExpert] = {
        "fight":    FightExpert(),
        "crash":    CrashExpert(),
        "fire":     fire_expert,
        "robbery":  RobberyExpert(checkpoint_path=str(ROBBERY_CKPT)),
        "carparts": CarPartsExpert(),
    }
    with _quiet():
        for exp in experts.values():
            exp.load(clip_model, clip_preprocess)

    with _quiet():
        tracker = ArconteTracker()

    logging.info("[W%d] listo", worker_id)

    while True:
        item = video_queue.get()
        if item is None:
            break

        video_path, expert_type = item

        # Checkpoint por video: si ya fue procesado anteriormente, reusar
        ckpt_file = _ckpt_path(video_path)
        if ckpt_file.exists():
            clip_results = np.load(ckpt_file, allow_pickle=True).tolist()
            results_queue.put((str(video_path), clip_results))
            logging.info("[W%d] SKIP (checkpoint) %s", worker_id, video_path.name)
            continue

        try:
            clip_results = _score_video(video_path, expert_type, experts, tracker)
            # Guardar checkpoint inmediatamente
            np.save(ckpt_file, clip_results, allow_pickle=True)
            results_queue.put((str(video_path), clip_results))
            n_det = sum(1 for c in clip_results if c["is_active"])
            logging.info("[W%d] %s → %d clips  |  %d DETECCIONES (%.0f%%)",
                         worker_id, video_path.name, len(clip_results),
                         n_det, 100 * n_det / max(len(clip_results), 1))
        except Exception:
            logging.exception("[W%d] Error: %s", worker_id, video_path.name)
            results_queue.put((str(video_path), []))

    results_queue.put(None)  # sentinel


# ────────────────────────────────────────────────────────────────────────────
# Step 2 — Distribución normal (C2FPL compatible)
# ────────────────────────────────────────────────────────────────────────────

def build_normal_distribution(
    vmae_dir:     Path,
    n_components: int = 5,
) -> Tuple[float, float]:
    """
    Carga features normales, computa norma L2 por clip, ajusta Gaussiana.
    Retorna (mu, sigma) para cómputo de p-valores.
    Compatible con C2FPL (usa norma L2 como señal escalar).
    """
    normal_dir = vmae_dir / "normal"
    npy_files  = sorted(normal_dir.glob("*.npy"))

    if not npy_files:
        raise FileNotFoundError(f"No features normales en {normal_dir}")

    logging.info("Cargando %d features normales para distribución...", len(npy_files))
    l2_norms = np.array([
        float(np.linalg.norm(np.load(f))) for f in npy_files
    ], dtype=np.float32)

    mu    = float(np.mean(l2_norms))
    sigma = float(np.std(l2_norms))
    logging.info("Distribución normal: mu=%.4f  sigma=%.4f", mu, sigma)
    return mu, sigma


def _pvalue(l2: float, mu: float, sigma: float) -> float:
    """P-valor two-tailed: qué tan probable es este clip bajo la distribución normal."""
    z = abs(l2 - mu) / max(sigma, 1e-8)
    return float(2.0 * (1.0 - scipy_norm.cdf(z)))


# ────────────────────────────────────────────────────────────────────────────
# Sliding window — réplica exacta C2FPL
# ────────────────────────────────────────────────────────────────────────────

def sliding_window_mask(pvalues: np.ndarray, window_ratio: float = 0.2) -> np.ndarray:
    """
    Encuentra el segmento contiguo de 'window_ratio * N' clips con mayor
    variación en p-valores → ese segmento es candidato a contener la anomalía.
    Retorna máscara binaria float32 (1.0 = en ventana, 0.0 = fuera).
    """
    n    = len(pvalues)
    mask = np.zeros(n, dtype=np.float32)
    wlen = max(1, int(n * window_ratio))

    if n <= wlen:
        mask[:] = 1.0
        return mask

    variations = []
    for i in range(n - wlen + 1):
        var = sum(abs(pvalues[j+1] - pvalues[j]) for j in range(i, i + wlen - 1))
        variations.append(var)

    best = int(np.argmax(variations))
    mask[best: best + wlen] = 1.0
    return mask


# ────────────────────────────────────────────────────────────────────────────
# Step 4 — Visual clips MP4
# ────────────────────────────────────────────────────────────────────────────

def save_visual_clip(
    video_path: Path,
    start:      int,
    score:      float,
    cat:        str,
    p_val:      float,
    out_path:   Path,
) -> None:
    """
    Guarda los CLIP_LEN frames (stride original) como MP4 con overlay de
    categoría, score y p-valor para verificación visual.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_idx = [start + i * STRIDE for i in range(CLIP_LEN)]

    if src_idx[-1] >= total:
        cap.release()
        return

    # Leer solo los frames necesarios
    frames: Dict[int, np.ndarray] = {}
    fi = 0
    needed = set(src_idx)
    while fi <= src_idx[-1]:
        ret, frm = cap.read()
        if not ret:
            break
        if fi in needed:
            frames[fi] = frm
        fi += 1
    cap.release()

    if len(frames) < CLIP_LEN:
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 8.0, (fw, fh))
    color  = CLASS_COLORS.get(cat, (255, 255, 255))

    for idx in src_idx:
        frm = frames[idx].copy()
        cv2.rectangle(frm, (0, 0), (fw, 52), (0, 0, 0), -1)
        cv2.putText(
            frm,
            f"{cat.upper()}  score={score:.3f}  p-val={p_val:.4f}",
            (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA,
        )
        writer.write(frm)

    writer.release()


# ────────────────────────────────────────────────────────────────────────────
# Step 3 — GT Generation
# ────────────────────────────────────────────────────────────────────────────

def generate_gt(
    all_scores:   Dict[str, List[Dict[str, Any]]],
    items:        List[Tuple[Path, str]],
    vmae_dir:     Path,
    out_dir:      Path,
    mu:           float,
    sigma:        float,
    p_threshold:  float,
    window_ratio: float,
    save_visual:  bool,
) -> None:
    """
    Para cada video aplica:
      · p-valores por clip (norma L2 vs Gaussiana normal)
      · sliding window → segmento más anómalo
      · intersección con expert.is_active + score

    Reglas de etiquetado
    --------------------
      Normal video      → label=0.0, class=0  (siempre)
      Anómalo, en ventana de señal combinada → label=score_norm (soft 0–1), class=CLASS_ID
      Anómalo, fuera de ventana              → label=0.0, class=0

    Señal combinada = score_norm * (1 - p_value)  [AND suave]
      · score_norm  : score del experto normalizado [0,1] dentro del video
      · (1-p_value) : qué tan lejos está el clip de la distribución normal
      Producto: alto solo cuando ambas señales coinciden → menos falsos positivos
    Sliding window sobre señal combinada → segmento de mayor variación temporal.

    Nota: is_active NO se usa — es unreliable en modo clip (requiere video continuo).
    El score raw se normaliza por video, por lo que la escala del experto no importa.

    Salidas C2FPL-compatibles: gt_binary, gt_multiclass, nalist, clip_names
    """
    visual_dir = out_dir / "visual_clips"

    gt_binary:     List[float] = []
    gt_multiclass: List[int]   = []
    nalist:        List[Tuple[int, int]] = []
    clip_names:    List[str]   = []
    cursor = 0

    stats = {"confirmed": 0, "normal": 0, "missing_feat": 0, "discarded": 0}

    for video_path, expert_type in items:
        video_key    = str(video_path)
        clip_results = all_scores.get(video_key, [])
        if not clip_results:
            continue

        vmae_cat_dir = vmae_dir / expert_type

        # ── Cargar features VideoMAE + calcular p-valores por clip ────────────
        clip_data: List[Dict[str, Any]] = []
        for cd in clip_results:
            feat_path = vmae_cat_dir / f"{cd['clip_name']}.npy"
            if not feat_path.exists():
                stats["missing_feat"] += 1
                continue
            l2    = float(np.linalg.norm(np.load(feat_path)))
            p_val = _pvalue(l2, mu, sigma)
            clip_data.append({**cd, "p_val": p_val})

        if not clip_data:
            continue

        # ── Normalizar scores del experto [0,1] dentro del video ─────────────
        raw_scores = np.array([c["score"] for c in clip_data], dtype=np.float32)
        s_min, s_max = raw_scores.min(), raw_scores.max()
        if s_max > s_min:
            scores_norm = (raw_scores - s_min) / (s_max - s_min)
        else:
            scores_norm = np.zeros_like(raw_scores)  # video plano → sin señal

        # ── Señal combinada: producto experto × VideoMAE (AND suave) ────────────
        pvalues          = np.array([c["p_val"] for c in clip_data], dtype=np.float32)
        anomaly_vmae     = 1.0 - pvalues                  # p bajo = anómalo = señal alta
        combined_signal  = scores_norm * anomaly_vmae     # alto solo si ambas coinciden

        # ── Sliding window sobre señal combinada (C2FPL style) ───────────────
        sw_mask = sliding_window_mask(combined_signal, window_ratio)

        # ── Etiquetar cada clip ───────────────────────────────────────────────
        v_from = cursor

        for i, cd in enumerate(clip_data):
            in_window = bool(sw_mask[i] > 0)

            if expert_type == "normal":
                label    = 0.0
                class_id = CLASS_ID["normal"]
                stats["normal"] += 1

            else:
                if in_window:
                    # Soft label = score normalizado del experto en este clip
                    label    = float(scores_norm[i])
                    class_id = CLASS_ID[expert_type]
                    stats["confirmed"] += 1

                    if save_visual:
                        mp4_out = visual_dir / expert_type / f"{cd['clip_name']}.mp4"
                        save_visual_clip(
                            video_path, cd["start"], float(scores_norm[i]),
                            expert_type, cd["p_val"], mp4_out
                        )
                else:
                    label    = 0.0
                    class_id = CLASS_ID["normal"]
                    stats["normal"] += 1

            gt_binary.append(label)
            gt_multiclass.append(class_id)
            clip_names.append(cd["clip_name"])
            cursor += 1

        v_to = cursor
        if v_to > v_from:
            nalist.append((v_from, v_to))

    # ── Guardar ───────────────────────────────────────────────────────────────
    np.save(out_dir / "gt_binary.npy",
            np.array(gt_binary,     dtype=np.float32))
    np.save(out_dir / "gt_multiclass.npy",
            np.array(gt_multiclass, dtype=np.int32))
    np.save(out_dir / "nalist.npy",
            np.array(nalist,        dtype=np.int32))
    np.save(out_dir / "clip_names.npy",
            np.array(clip_names,    dtype=object), allow_pickle=True)

    n_anom = sum(1 for l in gt_binary if l > 0.0)
    n_norm = sum(1 for l in gt_binary if l == 0.0)

    logging.info("=" * 72)
    logging.info("GT generado:")
    logging.info("  Total clips   : %d", len(gt_binary))
    logging.info("  Anómalos soft : %d  (ambas señales coinciden)", stats["confirmed"])
    logging.info("  Normales      : %d", n_norm)
    logging.info("  Descartados   : %d  (inciertos)", stats["discarded"])
    logging.info("  Sin feature   : %d  (VideoMAE no generado)", stats["missing_feat"])
    logging.info("  Clases: %s", {k: sum(1 for c in gt_multiclass if c == v)
                                   for k, v in CLASS_ID.items()})
    logging.info("=" * 72)


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genera pseudo-labels para VideoMAE fine-tuning"
    )
    p.add_argument("--txt",          type=Path,  default=DATASET_TXT)
    p.add_argument("--video-root",   type=Path,  default=VIDEO_ROOT)
    p.add_argument("--vmae-dir",     type=Path,  default=VMAE_FEATURES)
    p.add_argument("--output",       type=Path,  default=OUTPUT_PATH)
    p.add_argument("--workers",      type=int,   default=NUM_WORKERS)
    p.add_argument("--max-vram",     type=float, default=MAX_VRAM_GB)
    p.add_argument("--window",       type=float, default=WINDOW_RATIO,
                   help="Tamaño ventana C2FPL como fracción del video (default 0.2)")
    p.add_argument("--p-threshold",  type=float, default=P_THRESHOLD,
                   help="P-valor < N → estadísticamente anómalo (default 0.05)")
    p.add_argument("--gmm-k",        type=int,   default=GMM_COMPONENTS)
    p.add_argument("--no-visual",    action="store_true",
                   help="No guardar MP4 de verificación visual")
    p.add_argument("--skip-scoring", action="store_true",
                   help="Saltar Step 1 si expert_scores.npy ya existe")
    p.add_argument("--limit",        type=int,   default=0,
                   help="Procesar máximo N videos (0 = todos)")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Logging
    fmt      = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh       = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    fh       = logging.FileHandler(args.output / "generator.log", encoding="utf-8")
    fh.setFormatter(fmt)
    root     = logging.getLogger()
    root.addHandler(sh)
    root.addHandler(fh)
    root.setLevel(logging.INFO)

    scores_cache = args.output / "expert_scores.npy"

    # ── Build video list ──────────────────────────────────────────────────────
    with open(args.txt, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    items: List[Tuple[Path, str]] = []
    for line in lines:
        parts = line.split("/", 1)
        if len(parts) < 2:
            continue
        expert_type = CATEGORY_MAP.get(parts[0])
        if expert_type is None:
            continue
        items.append((args.video_root / line, expert_type))

    if args.limit:
        items = items[: args.limit]

    logging.info("=" * 72)
    logging.info("Pseudo-Label Generator  |  UCF-Crime")
    logging.info("  Videos       : %d", len(items))
    logging.info("  Workers      : %d", args.workers)
    logging.info("  P-threshold  : %.3f", args.p_threshold)
    logging.info("  Window ratio : %.0f%%", args.window * 100)
    logging.info("  Visual clips : %s", not args.no_visual)
    logging.info("=" * 72)

    # ── Step 1: Expert Scoring ────────────────────────────────────────────────
    all_scores: Dict[str, List[Dict[str, Any]]] = {}

    if args.skip_scoring and scores_cache.exists():
        logging.info("Step 1: cargando expert scores desde caché (%s)...", scores_cache)
        all_scores = np.load(scores_cache, allow_pickle=True).item()
    else:
        logging.info("Step 1: expert scoring con %d workers...", args.workers)
        t0 = time.perf_counter()

        log_queue:     mp.Queue = mp.Queue(-1)
        video_queue:   mp.Queue = mp.Queue()
        results_queue: mp.Queue = mp.Queue()

        # Listener de logging
        log_fh = logging.FileHandler(args.output / "scoring.log", encoding="utf-8")
        log_fh.setFormatter(fmt)
        listener = logging.handlers.QueueListener(log_queue, log_fh, sh)
        listener.start()

        # Solo encolar videos ANÓMALOS — normales siempre label=0.0, no necesitan scoring
        anomaly_items = [(vp, et) for vp, et in items if et != "normal"]
        logging.info("  Videos a scorear : %d  (normales saltados: %d)",
                     len(anomaly_items), len(items) - len(anomaly_items))

        for video_path, expert_type in anomaly_items:
            video_queue.put((video_path, expert_type))
        for _ in range(args.workers):
            video_queue.put(None)

        workers = []
        for wid in range(args.workers):
            p = mp.Process(
                target=_worker_main,
                name=f"ScoreW{wid}",
                args=(wid, video_queue, results_queue, log_queue,
                      args.max_vram, args.workers),
            )
            p.start()
            workers.append(p)

        sentinels = 0
        while sentinels < args.workers:
            item = results_queue.get()
            if item is None:
                sentinels += 1
                continue
            video_path_str, clip_results = item
            all_scores[video_path_str] = clip_results

        for p in workers:
            p.join()
        listener.stop()

        np.save(scores_cache, all_scores, allow_pickle=True)
        logging.info("Step 1: done en %.0f min — caché en %s",
                     (time.perf_counter() - t0) / 60, scores_cache)

    # ── Step 2: Distribución normal ───────────────────────────────────────────
    logging.info("Step 2: ajustando distribución normal sobre features VideoMAE...")
    mu, sigma = build_normal_distribution(args.vmae_dir, args.gmm_k)

    # ── Step 3 + 4: GT + Visual clips ────────────────────────────────────────
    logging.info("Step 3: generando GT (binario + multiclase)...")
    generate_gt(
        all_scores   = all_scores,
        items        = items,
        vmae_dir     = args.vmae_dir,
        out_dir      = args.output,
        mu           = mu,
        sigma        = sigma,
        p_threshold  = args.p_threshold,
        window_ratio = args.window,
        save_visual  = not args.no_visual,
    )

    logging.info("DONE — outputs en %s", args.output)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
