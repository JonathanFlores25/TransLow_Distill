"""
dataset.py
==========
Dataset para VideoMAE VAD Classifier.
Estructura basada en Step2_Detector/dataset.py adaptada para features VideoMAE.

Diferencias respecto a Step2 (X3D):
- Las features no están pre-concatenadas en un solo .npy sino en archivos
  individuales por clip: split_features/{split}/{category}/{stem}_s{start}.npy
  Cada archivo es un vector (768,) float32.
- Las pseudo-labels vienen de pseudo_label_generator.py en formato flat:
    args.clip_names_path → clip_names.npy   (N,) nombres de clips
    args.gt_binary_path  → gt_binary.npy    (N,) soft labels 0.0-1.0
  Se binariza con args.pseudo_threshold (default 0.5).
  Clips normales no presentes en el lookup conservan label=0.
- Para test/evaluación, los clips exponen su start frame (del nombre de archivo)
  para que metrics_library.clips_to_frames() funcione correctamente.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Sampler

_NORMAL_DIR = "normal"
_SPLIT_MAP  = {"Train": "train", "Validacion": "val", "Test": "test"}
_STEM_RE    = re.compile(r"^(.+)_s(\d{6})$")   # parses  VideoName_x264_s002256


# ─────────────────────────────────────────────────────────────────────────────
# Balanced sampler  (igual que Step2, con fallback para subsets pequeños)
# ─────────────────────────────────────────────────────────────────────────────

class BalancedBatchSampler(Sampler):
    """
    Genera batches con pos_fraction anomalías y (1-pos_fraction) normales.
    Replica la lógica de Step2_Detector con fallback en lugar de ValueError.
    """

    def __init__(self, labels, batch_size, pos_fraction=0.5, seed=0):
        self.labels   = np.asarray(labels).astype(int)
        self.batch_size = batch_size
        self.pos_bs   = int(round(batch_size * pos_fraction))
        self.neg_bs   = batch_size - self.pos_bs
        self.rng      = np.random.default_rng(seed)

        self.pos_idx = np.where(self.labels == 1)[0]
        self.neg_idx = np.where(self.labels == 0)[0]

        if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
            warnings.warn(
                f"BalancedBatchSampler: {len(self.pos_idx)} positivos / "
                f"{len(self.neg_idx)} negativos. Usando permutación simple.",
                stacklevel=2,
            )
            self._fallback   = True
            self.num_batches = int(np.ceil(len(self.labels) / batch_size))
        else:
            self._fallback   = False
            self.num_batches = int(np.ceil(len(self.neg_idx) / self.neg_bs))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        if self._fallback:
            perm = self.rng.permutation(len(self.labels))
            for s in range(0, len(self.labels), self.batch_size):
                yield perm[s : s + self.batch_size].tolist()
            return

        neg_perm = self.rng.permutation(self.neg_idx)
        neg_ptr  = 0

        for _ in range(self.num_batches):
            if neg_ptr + self.neg_bs <= len(neg_perm):
                neg_batch = neg_perm[neg_ptr : neg_ptr + self.neg_bs]
                neg_ptr  += self.neg_bs
            else:
                remaining = len(neg_perm) - neg_ptr
                neg_batch = np.concatenate([
                    neg_perm[neg_ptr:],
                    self.rng.choice(self.neg_idx, self.neg_bs - remaining, replace=True),
                ])
                neg_ptr = len(neg_perm)

            pos_batch = self.rng.choice(self.pos_idx, self.pos_bs, replace=True)
            batch     = np.concatenate([neg_batch, pos_batch])
            self.rng.shuffle(batch)
            yield batch.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset principal
# ─────────────────────────────────────────────────────────────────────────────

class Dataset_VideoMAE(data.Dataset):
    """
    Carga features VideoMAE (768,) por clip para train / val / test.

    Uso (igual que Step2):
        dataset    = Dataset_VideoMAE(args, test_Mode="Train")
        labels     = np.asarray(dataset.labels_all).astype(int)
        train_loader = DataLoader(dataset,
                           batch_sampler=BalancedBatchSampler(labels, batch_size=64),
                           num_workers=4)

        test_loader  = DataLoader(Dataset_VideoMAE(args, test_Mode="Test"),
                           batch_size=256, shuffle=False)

    args requeridos:
        args.features_dir      — raíz de split_features/ (Path o str)
        args.clip_names_path   — data/pseudo_labels/clip_names.npy
        args.gt_binary_path    — data/pseudo_labels/gt_binary.npy
        args.pseudo_threshold  — umbral binarización (default 0.5)

    Atributos expuestos (necesarios para metrics_library):
        self.labels_all    — list[int]  etiquetas efectivas (train) o dir-label (test)
        self.file_paths    — list[Path] ruta de cada clip .npy
        self.video_stems   — list[str]  nombre del video sin _s{start}
        self.clip_starts   — list[int]  frame de inicio del clip (del nombre)
        self.categories    — list[str]  categoría del subdirectorio
    """

    def __init__(self, args, test_Mode: str = ""):
        self.test_mode = test_Mode
        split          = _SPLIT_MAP.get(test_Mode, "train")
        split_dir      = Path(args.features_dir) / split

        # Fallback: si no existe el subdirectorio de split, usar la raíz directamente.
        # Esto permite apuntar features_dir a un directorio plano {categoria}/*.npy
        # sin necesidad de crear subdirectorios train/val/test.
        if not split_dir.exists():
            split_dir = Path(args.features_dir)
            if not split_dir.exists():
                raise FileNotFoundError(
                    f"No se encontró el directorio de features: {split_dir}\n"
                    f"Verifica args.features_dir = {args.features_dir}"
                )
            warnings.warn(
                f"Subdirectorio '{split}' no encontrado en {args.features_dir}. "
                f"Usando el directorio raíz como fallback (estructura plana).",
                stacklevel=2,
            )

        # ── Descubrir todos los clips del split ───────────────────────────────
        self.file_paths:  list[Path] = []
        self.labels_all:  list[int]  = []   # label de directorio (base)
        self.video_stems: list[str]  = []   # e.g. "Assault001_x264"
        self.clip_starts: list[int]  = []   # e.g. 2256
        self.categories:  list[str]  = []   # e.g. "fight"

        for npy_path in sorted(split_dir.rglob("*.npy")):
            category  = npy_path.parent.name
            dir_label = 0 if category == _NORMAL_DIR else 1

            m = _STEM_RE.match(npy_path.stem)
            if m:
                video_stem  = m.group(1)
                clip_start  = int(m.group(2))
            else:
                video_stem  = npy_path.stem
                clip_start  = 0

            self.file_paths.append(npy_path)
            self.labels_all.append(dir_label)
            self.video_stems.append(video_stem)
            self.clip_starts.append(clip_start)
            self.categories.append(category)

        if len(self.file_paths) == 0:
            raise FileNotFoundError(
                f"No se encontraron archivos .npy en {split_dir}.\n"
                f"Ejecuta FE_VideoMAE.py primero para extraer features."
            )

        n_pos = sum(self.labels_all)
        n_neg = len(self.labels_all) - n_pos
        print(
            f"[Dataset_VideoMAE] {test_Mode or 'Train'}:  "
            f"total={len(self.file_paths)}  normal={n_neg}  anomalía={n_pos}"
        )

        # ── Pseudo-labels (solo en entrenamiento) ─────────────────────────────
        if test_Mode == "Train":
            self._apply_pseudo_labels(args)

    # ── Pseudo-label lookup desde flat arrays ─────────────────────────────────

    def _apply_pseudo_labels(self, args) -> None:
        """
        Reemplaza las etiquetas de directorio con las pseudo-labels de
        pseudo_label_generator.py (clip_names.npy + gt_binary.npy).

        Regla:
          - Clip presente en lookup AND gt_binary > threshold → label=1
          - Clip presente en lookup AND gt_binary <= threshold → label=0
          - Clip ausente del lookup (e.g. clips normales) → conserva label=0
        """
        names_path = Path(args.clip_names_path)
        gt_path    = Path(args.gt_binary_path)

        if not names_path.exists() or not gt_path.exists():
            warnings.warn(
                f"Pseudo-labels no encontradas:\n  {names_path}\n  {gt_path}\n"
                f"Usando etiquetas de directorio (coarse binary).",
                stacklevel=2,
            )
            return

        threshold  = float(getattr(args, "pseudo_threshold", 0.5))
        clip_names = np.load(names_path, allow_pickle=True)
        gt_binary  = np.load(gt_path).astype(np.float32)

        if len(clip_names) != len(gt_binary):
            raise ValueError(
                f"Mismatch pseudo-labels: clip_names={len(clip_names)} "
                f"vs gt_binary={len(gt_binary)}. "
                f"Vuelve a correr pseudo_label_generator.py."
            )

        lookup = {
            str(name): int(float(score) > threshold)
            for name, score in zip(clip_names, gt_binary)
        }

        n_changed = 0
        for i, fp in enumerate(self.file_paths):
            if fp.stem in lookup:
                new_label = lookup[fp.stem]
                if new_label != self.labels_all[i]:
                    n_changed += 1
                self.labels_all[i] = new_label

        n_pos = sum(self.labels_all)
        print(
            f"[Dataset_VideoMAE] Pseudo-labels aplicadas: "
            f"{len(lookup)} clips en lookup | "
            f"{n_changed} etiquetas cambiadas | "
            f"{n_pos} positivos ({100*n_pos/max(len(self.labels_all),1):.1f}%)"
        )

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int):
        """
        Train      → (features, label)   features: (768,) float32
        Test / Val → features             para inferencia con metrics_library
        """
        feat = np.load(self.file_paths[index]).astype(np.float32)
        features = torch.tensor(feat, dtype=torch.float32)

        if self.test_mode in ["Test", "Validacion"]:
            return features
        else:
            label = torch.tensor(self.labels_all[index], dtype=torch.float32)
            return features, label
