"""
dataset.py
==========
Dataset infrastructure for VideoMAE VAD Classifier (Step3).

Design rationale vs Step2_Detector:
- Per-file .npy discovery instead of pre-concatenated arrays: FE_VideoMAE.py
  stores one (768,) array per clip in category subdirectories. Loading them
  individually avoids the memory spike of concatenating thousands of clips
  into a single array at startup, and preserves the per-video filename
  information needed for frame-level score reconstruction at test time.
- Binary label from subdirectory name: any subdirectory whose name is NOT
  "normal" is treated as anomaly (label=1). This mirrors CATEGORY_MAP in
  FE_VideoMAE.py where all non-normal categories map to expert subdirs.
- z-score normalization via stats dict: VideoMAE was pre-trained on
  Kinetics-400. UCF-Crime surveillance features occupy a different region
  of the 768-dim space. Per-feature standardization (mean/std from training
  split only, Welford online algorithm) reduces this domain shift without
  requiring whitening or PCA, which would destroy the spatiotemporal
  entanglement structure of the tube embeddings.
- Welford algorithm in compute_normalization_stats: avoids loading all N
  features into RAM to compute mean/std; processes one sample at a time with
  numerically stable running variance accumulation.
- BalancedBatchSampler graceful zero-class handling: Step2 raised a
  hard ValueError when one class was absent. Here we fall back to standard
  random sampling with a warning, which prevents crashes during debugging
  with small subset datasets.
- collate_fn enforces float32: ensures no accidental float64 promotion when
  numpy loads .npy files with varying dtypes.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# The feature dimension is fixed by VideoMAE ViT-Base architecture.
# 12 Transformer layers × 12 heads × 64 head_dim = 768.
# This constant is declared here so assert messages reference it directly.
_VMAE_FEATURE_DIM = 768

# Subdirectory name that signals normal (non-anomaly) clips.
# All other category subdirectories (fight, crash, fire, robbery, carparts)
# are treated as anomalous — this mirrors CATEGORY_MAP in FE_VideoMAE.py.
_NORMAL_DIR_NAME = "normal"


class VideoMAEFeatureDataset(Dataset):
    """
    Discovers and serves pre-extracted VideoMAE clip features.

    Expected on-disk layout (produced by FE_VideoMAE.py)::

        root_dir/
          train/
            normal/
              SomeFightingVideo_s000000.npy   ← (768,) float32
              ...
            fight/
              FightingVideo001_s000000.npy
              ...
          val/
            ...
          test/
            ...

    Each .npy file is a (768,) float32 array representing the
    last_hidden_state.mean(dim=1) over all 1568 patch tokens of one 16-frame
    clip.

    Parameters
    ----------
    root_dir : Path
        Root features directory (the parent of train/val/test splits).
    split : str
        One of 'train', 'val', 'test'.
    pseudo_label_dir : Path, optional
        If provided, binary labels are loaded from matching .npy files in
        this directory (C2FPL weakly-supervised curriculum labels).
        Files must follow the same naming convention as features.
        If a pseudo-label file is missing for a clip, the subdirectory-derived
        label is used as fallback.
    stats : dict, optional
        Normalization statistics with keys 'mean' and 'std', each a
        (768,) float32 np.ndarray. When provided, features are z-scored as
        (feat - mean) / (std + 1e-8) before being returned. Must be computed
        on the TRAINING split only and then passed unchanged to val/test.
    """

    def __init__(
        self,
        root_dir: Path,
        split: str,
        pseudo_label_dir: Optional[Path] = None,
        stats: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.pseudo_label_dir = Path(pseudo_label_dir) if pseudo_label_dir else None
        self.stats = stats

        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. "
                f"Expected features_dir/{split}/ to contain category subdirectories "
                f"produced by FE_VideoMAE.py."
            )

        # Collect all .npy paths and derive their binary labels.
        # Sorting ensures deterministic ordering across file systems, which
        # matters for reproducibility of BalancedBatchSampler indices.
        self.file_paths: List[Path] = []
        self.labels: List[int] = []

        for npy_path in sorted(split_dir.rglob("*.npy")):
            # The immediate parent of each .npy is the category subdirectory.
            # Any name other than "normal" → anomaly (label=1).
            category = npy_path.parent.name
            label = 0 if category == _NORMAL_DIR_NAME else 1
            self.file_paths.append(npy_path)
            self.labels.append(label)

        if len(self.file_paths) == 0:
            raise FileNotFoundError(
                f"No .npy files found under {split_dir}. "
                f"Run FE_VideoMAE.py first to extract VideoMAE features."
            )

        # Build pseudo-label lookup if directory provided.
        # Maps stem (without .npy extension) → pseudo label scalar.
        # Using a dict keyed by relative path stem avoids path-separator issues.
        self._pseudo_lookup: Dict[str, int] = {}
        if self.pseudo_label_dir is not None and self.pseudo_label_dir.exists():
            for pl_path in self.pseudo_label_dir.rglob("*.npy"):
                arr = np.load(pl_path)
                # Pseudo-label files contain a single scalar (0 or 1).
                label_val = int(np.round(float(arr.flat[0])))
                self._pseudo_lookup[pl_path.stem] = label_val

        n_pos = sum(self.labels)
        n_neg = len(self.labels) - n_pos
        print(
            f"[VideoMAEFeatureDataset] split={split}  "
            f"total={len(self.file_paths)}  normal={n_neg}  anomaly={n_pos}"
        )

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns
        -------
        feature : torch.Tensor of shape (768,) dtype float32
            z-scored feature if stats were provided, raw feature otherwise.
        label : int
            0 = Normal, 1 = Anomaly.
        """
        path = self.file_paths[index]
        feat = np.load(path)

        # Shape guard: every clip feature produced by FE_VideoMAE.py must be
        # exactly (768,) — the mean over 1568 VideoMAE patch tokens.
        assert feat.shape == (_VMAE_FEATURE_DIM,), (
            f"Feature shape mismatch: expected ({_VMAE_FEATURE_DIM},) but got "
            f"{feat.shape} for file: {path}. "
            f"Re-extract features with FE_VideoMAE.py if the backbone changed."
        )

        # Cast to float32 explicitly — numpy may produce float64 on some
        # file systems, and we must stay in float32 throughout to avoid
        # accidental promotion in loss computation.
        feat = feat.astype(np.float32)

        # Apply z-score normalization with training-split statistics.
        # The +1e-8 epsilon prevents division-by-zero for near-constant dims
        # that sometimes occur in under-represented feature channels.
        if self.stats is not None:
            mean = self.stats["mean"]  # (768,) float32
            std = self.stats["std"]    # (768,) float32
            feat = (feat - mean) / (std + 1e-8)

        # Resolve label: pseudo-label takes precedence over directory label
        # when a matching file exists in pseudo_label_dir.
        label = self.labels[index]
        stem = path.stem
        if stem in self._pseudo_lookup:
            label = self._pseudo_lookup[stem]

        return torch.tensor(feat, dtype=torch.float32), label

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def label_array(self) -> np.ndarray:
        """Return labels as int32 numpy array — needed by BalancedBatchSampler."""
        return np.array(self.labels, dtype=np.int32)


# ── Normalization statistics ──────────────────────────────────────────────────

def compute_normalization_stats(
    dataset: VideoMAEFeatureDataset,
) -> Dict[str, np.ndarray]:
    """
    Compute per-feature mean and standard deviation over a dataset using
    Welford's online algorithm.

    WHY Welford: loading all N clips into RAM to call np.mean/np.std would
    require N × 768 × 4 bytes ≈ 3 GB for 1M clips. Welford processes one
    sample at a time with O(1) memory and numerically stable accumulation.

    WHY training split only: using test-split statistics would leak information
    about the test distribution into feature normalization, creating an
    evaluation artifact. Stats must be computed once on training data and
    then applied identically to val and test.

    Parameters
    ----------
    dataset : VideoMAEFeatureDataset
        Must be the TRAINING split dataset, loaded WITHOUT stats (stats=None),
        so raw Kinetics-domain features are standardized.

    Returns
    -------
    dict with keys:
        'mean' : np.ndarray of shape (768,) float32
        'std'  : np.ndarray of shape (768,) float32
            Per-feature standard deviation. Channels with near-zero std
            (< 1e-6) are clipped to 1e-6 to prevent division by zero.
    """
    n = len(dataset)
    if n == 0:
        raise ValueError("Cannot compute normalization stats on an empty dataset.")

    # Welford running accumulators — kept in float64 internally to avoid
    # catastrophic cancellation when accumulating many small squared differences.
    # Final output is cast to float32 for storage.
    mean_acc = np.zeros(_VMAE_FEATURE_DIM, dtype=np.float64)
    M2_acc = np.zeros(_VMAE_FEATURE_DIM, dtype=np.float64)  # sum of squared diffs

    for i in range(n):
        feat_tensor, _ = dataset[i]
        x = feat_tensor.numpy().astype(np.float64)

        # Welford update: each new sample shifts the running mean and M2
        count = i + 1
        delta = x - mean_acc
        mean_acc += delta / count
        delta2 = x - mean_acc
        M2_acc += delta * delta2

        if i % 50000 == 0 and i > 0:
            print(f"  [compute_normalization_stats] {i}/{n} samples processed ...")

    # Sample variance (unbiased) — use max(n-1, 1) to avoid zero division on
    # single-sample edge case during unit tests.
    variance = M2_acc / max(n - 1, 1)
    std = np.sqrt(variance)

    # Clamp near-zero std to prevent explosive normalization on dead channels.
    std = np.clip(std, a_min=1e-6, a_max=None)

    return {
        "mean": mean_acc.astype(np.float32),
        "std": std.astype(np.float32),
    }


# ── Balanced sampler ──────────────────────────────────────────────────────────

class BalancedBatchSampler(Sampler):
    """
    Yields batches with exactly pos_fraction anomaly clips and
    (1-pos_fraction) normal clips.

    WHY: UCF-Crime has ~6:1 normal-to-anomaly ratio at clip level. Without
    balancing, the classifier learns to always predict normal. The balanced
    sampler interleaves oversampled anomaly clips with normal clips so the
    loss gradient is equally informed by both classes each step.

    Graceful zero-class fallback (fix vs Step2_Detector):
    Step2 raised ValueError when a class was absent, crashing DataLoader
    workers during debugging on subsets. Here we detect the condition,
    emit a warning, and fall back to random permutation — allowing the rest
    of the pipeline to proceed for smoke-testing.

    Parameters
    ----------
    labels : array-like of int
        Binary labels (0=normal, 1=anomaly) for every sample in the dataset,
        in dataset index order.
    batch_size : int
        Total clips per batch.
    pos_fraction : float
        Target fraction of anomaly samples per batch (default 0.5).
    seed : int
        RNG seed for reproducible shuffling.
    """

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        pos_fraction: float = 0.5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.labels = np.asarray(labels, dtype=np.int32)
        self.batch_size = batch_size
        self.pos_fraction = pos_fraction
        self.seed = seed

        self.pos_bs = max(1, int(round(batch_size * pos_fraction)))
        self.neg_bs = batch_size - self.pos_bs

        self.pos_idx = np.where(self.labels == 1)[0]
        self.neg_idx = np.where(self.labels == 0)[0]

        # Degenerate case: one class entirely absent.
        # This can happen with tiny debug subsets. Fall back gracefully.
        self._fallback = False
        if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
            warnings.warn(
                f"BalancedBatchSampler: split has {len(self.pos_idx)} anomaly "
                f"and {len(self.neg_idx)} normal samples. Falling back to "
                f"random permutation (no balancing). This is expected only for "
                f"debug subsets; a real training split must have both classes.",
                stacklevel=2,
            )
            self._fallback = True
            self.num_batches = int(np.ceil(len(self.labels) / batch_size))
        else:
            # Number of batches is driven by the negative (majority) class so
            # every normal clip is seen approximately once per epoch.
            # Anomaly clips are oversampled via replacement.
            self.num_batches = max(1, int(np.ceil(len(self.neg_idx) / self.neg_bs)))

        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        if self._fallback:
            # Fallback: yield simple random batches without class balancing.
            perm = self.rng.permutation(len(self.labels))
            for start in range(0, len(self.labels), self.batch_size):
                yield perm[start : start + self.batch_size].tolist()
            return

        # Shuffle the negative pool each epoch so different normals are paired
        # with oversampled anomalies. Reproducible via seeded rng.
        neg_perm = self.rng.permutation(self.neg_idx)
        neg_ptr = 0

        for _ in range(self.num_batches):
            # Fill negative slots, wrapping around with replacement when the
            # normal pool is exhausted (last batch of the epoch).
            if neg_ptr + self.neg_bs <= len(neg_perm):
                neg_batch = neg_perm[neg_ptr : neg_ptr + self.neg_bs]
                neg_ptr += self.neg_bs
            else:
                remaining = len(neg_perm) - neg_ptr
                extra = self.rng.choice(
                    self.neg_idx, self.neg_bs - remaining, replace=True
                )
                neg_batch = np.concatenate([neg_perm[neg_ptr:], extra])
                neg_ptr = len(neg_perm)

            # Anomaly clips are always oversampled with replacement because
            # there are fewer anomaly clips than normal clips in UCF-Crime.
            pos_batch = self.rng.choice(self.pos_idx, self.pos_bs, replace=True)

            batch = np.concatenate([neg_batch, pos_batch])
            self.rng.shuffle(batch)
            yield batch.tolist()


# ── Collate function ──────────────────────────────────────────────────────────

def collate_fn(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stack a list of (feature, label) pairs into tensors.

    Explicitly enforces float32 on both features and labels.
    - Features: [B, 768] float32 — ready for VideoMAE_VAD_Classifier.forward()
    - Labels: [B] float32 — required by BCELoss (not BCEWithLogitsLoss which
      would accept float as well, but keeping consistent dtype throughout).

    WHY float32 for labels: torch BCELoss expects target and input to have
    the same dtype. Using int64 labels would cause a silent upcasting issue
    on some PyTorch versions that produces NaN losses.
    """
    features = torch.stack([item[0] for item in batch], dim=0).to(torch.float32)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    return features, labels
