"""
sampler_v2.py
=============
Sampler de entrenamiento para VideoMAE VAD v2.

Diferencia respecto al BalancedBatchSampler de v1
--------------------------------------------------
v1 tiene un modo fallback: si una clase está ausente, hace permutación simple
y el batch puede ser 100% normal o 100% anómalo.

v2 usa GuaranteedBalancedSampler:
  · SIEMPRE garantiza min_pos positivos y min_neg negativos en cada batch,
    usando muestreo con reemplazo cuando el conjunto de una clase se agota.
  · Nunca aborta. Si hay muy pocos de una clase, sampleará los mismos
    índices repetidos antes que ceder un batch sin ambas clases.
  · Solo lanza ValueError si una clase tiene CERO muestras (imposible forzar).
  · El Soft Triplet OHNM necesita ≥2 positivos (anchor + hard positive)
    y ≥1 negativo (hard negative), de ahí min_pos=2, min_neg=1 por defecto.
"""

from __future__ import annotations

import warnings

import numpy as np
from torch.utils.data import Sampler


class GuaranteedBalancedSampler(Sampler):
    """
    Genera batches garantizando SIEMPRE pos_bs positivos y neg_bs negativos.

    Parámetros
    ----------
    labels       : array-like de int (0=normal, 1=anomalía)
    batch_size   : clips por batch
    pos_fraction : fracción de positivos por batch (default 0.5)
    min_pos      : positivos mínimos requeridos por batch (default 2)
    min_neg      : negativos mínimos requeridos por batch (default 1)
    seed         : semilla para reproducibilidad
    """

    def __init__(
        self,
        labels:       "array-like",
        batch_size:   int,
        pos_fraction: float = 0.5,
        min_pos:      int   = 2,
        min_neg:      int   = 1,
        seed:         int   = 0,
    ):
        self.labels     = np.asarray(labels).astype(int)
        self.batch_size = batch_size
        self.rng        = np.random.default_rng(seed)

        self.pos_idx = np.where(self.labels == 1)[0]
        self.neg_idx = np.where(self.labels == 0)[0]

        # ── Cero muestras: imposible garantizar — error claro ─────────────────
        if len(self.pos_idx) == 0:
            raise ValueError(
                "GuaranteedBalancedSampler: 0 positivos en el dataset.\n"
                "Verifica pseudo_label_generator.py y --pseudo_threshold."
            )
        if len(self.neg_idx) == 0:
            raise ValueError(
                "GuaranteedBalancedSampler: 0 negativos en el dataset.\n"
                "Verifica que features_dir contenga la carpeta 'normal'."
            )

        # ── Ajustar pos_bs para respetar los mínimos ──────��──────────────────
        # pos_bs = lo que pide pos_fraction, pero nunca < min_pos ni < 1
        self.pos_bs = max(min_pos, int(round(batch_size * pos_fraction)))
        # neg_bs llena el resto, pero nunca < min_neg ni < 1
        self.neg_bs = max(min_neg, batch_size - self.pos_bs)
        # Recalcular batch_size efectivo (puede diferir del pedido)
        self._effective_bs = self.pos_bs + self.neg_bs

        # Avisar si hay muy pocos de una clase (se reutilizarán índices)
        if len(self.pos_idx) < self.pos_bs:
            warnings.warn(
                f"GuaranteedBalancedSampler: solo {len(self.pos_idx)} positivos "
                f"para pos_bs={self.pos_bs}. Se muestreará con reemplazo.",
                stacklevel=2,
            )
        if len(self.neg_idx) < self.neg_bs:
            warnings.warn(
                f"GuaranteedBalancedSampler: solo {len(self.neg_idx)} negativos "
                f"para neg_bs={self.neg_bs}. Se muestreará con reemplazo.",
                stacklevel=2,
            )

        # Número de batches: cubre la clase más grande una vez por epoch
        n_larger = max(len(self.pos_idx), len(self.neg_idx))
        bs_larger = self.pos_bs if len(self.pos_idx) >= len(self.neg_idx) else self.neg_bs
        self.num_batches = int(np.ceil(n_larger / bs_larger))

        print(
            f"[GuaranteedBalancedSampler] "
            f"{len(self.pos_idx)} pos | {len(self.neg_idx)} neg | "
            f"{self.pos_bs} pos/batch | {self.neg_bs} neg/batch | "
            f"{self.num_batches} batches/epoch  "
            f"(batch_size efectivo={self._effective_bs})"
        )

    def __len__(self) -> int:
        return self.num_batches

    def _sample_class(self, idx_pool: np.ndarray, n: int, perm: np.ndarray, ptr: int):
        """
        Extrae n índices del pool. Si se agota la permutación, completa con
        muestreo aleatorio con reemplazo — NUNCA devuelve menos de n índices.
        Devuelve (batch_indices, nueva_permutación, nuevo_ptr).
        """
        if ptr + n <= len(perm):
            batch = perm[ptr : ptr + n]
            return batch, perm, ptr + n

        # Agotado: tomamos lo que queda y rellenamos
        remaining = perm[ptr:]
        needed    = n - len(remaining)
        # Nueva permutación para el próximo ciclo
        new_perm  = self.rng.permutation(idx_pool)
        fill      = new_perm[:needed]
        batch     = np.concatenate([remaining, fill])
        return batch, new_perm, needed

    def __iter__(self):
        neg_perm = self.rng.permutation(self.neg_idx)
        pos_perm = self.rng.permutation(self.pos_idx)
        neg_ptr = pos_ptr = 0

        for _ in range(self.num_batches):
            neg_batch, neg_perm, neg_ptr = self._sample_class(
                self.neg_idx, self.neg_bs, neg_perm, neg_ptr
            )
            pos_batch, pos_perm, pos_ptr = self._sample_class(
                self.pos_idx, self.pos_bs, pos_perm, pos_ptr
            )

            batch = np.concatenate([neg_batch, pos_batch])
            self.rng.shuffle(batch)
            yield batch.tolist()
