"""
train_v2.py
===========
Funciones de entrenamiento para VideoMAE VAD Classifier v2.

Loss combinada
--------------
  L_total = L_bce_smooth  +  λ_t · L_soft_triplet

  L_bce_smooth:
      BCE con Label Smoothing (ε=0.1) — robusto a pseudo-labels ruidosas.
      target_suavizado = label * (1 − ε) + ε / 2

  L_soft_triplet (Soft Triplet con OHNM):
      log(1 + exp( (d_ap² − d_an²) / τ ))
      · Sin zona muerta — gradiente continuo incluso en pares ya resueltos.
      · τ controla la dureza: τ pequeño → concentra el gradiente en casos
        difíciles; τ grande → distribuye el gradiente (más robusto a ruido).
      · OHNM: Hard Positive  = anomalía más LEJANA al anchor
              Hard Negative  = normal más CERCANA al anchor
      · Solo se computa sobre anchors con al menos 1 positivo y 1 negativo
        en el batch (skip graceful si el batch es demasiado homogéneo).

Compatibilidad con Fases 2 y 3
--------------------------------
  Las funciones de loss son independientes del modelo: reciben embeddings y
  labels como tensores. En Fase 2 (backbone semi-descongelado) y Fase 3
  (atención guiada), solo cambia qué genera los embeddings — la loss no.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# BCE con Label Smoothing
# ─────────────────────────────────────────────────────────────────────────────

class LabelSmoothingBCE(nn.Module):
    """
    BCE con suavizado de etiquetas.

    target_suavizado = label * (1 − smoothing) + smoothing / 2

    Con smoothing=0.1:
      · label=1 → target=0.95   (evita que el modelo sea demasiado confiado)
      · label=0 → target=0.05   (tolera ruido en clips normales)

    Esto es especialmente útil con pseudo-labels de UCF-Crime donde ~15-20%
    de las etiquetas pueden ser incorrectas.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_smooth = target * (1.0 - self.smoothing) + self.smoothing * 0.5
        return F.binary_cross_entropy(pred, target_smooth)


# ─────────────────────────────────────────────────────────────────────────────
# Soft Triplet Loss con OHNM
# ─────────────────────────────────────────────────────────────────────────────

def _pairwise_squared_dist(x: torch.Tensor) -> torch.Tensor:
    """
    Distancias euclidianas al cuadrado entre todos los pares del batch.

    Para embeddings L2-normalizados: d² = 2 − 2·cos(θ) ∈ [0, 4].
    Uso de la identidad ||a-b||² = ||a||² + ||b||² - 2·aᵀb para estabilidad.

    Input  : (B, D) float32
    Output : (B, B) float32
    """
    dot   = x @ x.t()                                  # (B, B)
    sq    = (x * x).sum(dim=1, keepdim=True)            # (B, 1)
    dist2 = sq + sq.t() - 2.0 * dot
    return dist2.clamp(min=0.0)                         # numérico: evita negativo


def soft_triplet_ohnm(
    embeddings: torch.Tensor,
    labels:     torch.Tensor,
    tau:        float = 0.5,
) -> torch.Tensor:
    """
    Soft Triplet Loss con Online Hard Negative Mining.

    Para cada anchor (anomalía):
      · Hard Positive  = anomalía con mayor distancia al anchor  (max d_ap)
      · Hard Negative  = normal  con menor distancia al anchor   (min d_an)
      · L_i = log(1 + exp( (d_ap − d_an) / τ ))

    Propiedades clave:
      · Sin zona muerta: siempre hay gradiente, incluso cuando d_an >> d_ap.
      · τ pequeño → gradiente concentrado en hard cases (agresivo).
      · τ grande  → gradiente distribuido (robusto a ruido en pseudo-labels).
      · Si un batch no tiene al menos 1 anomalía + 1 normal → devuelve 0.

    Parameters
    ----------
    embeddings : (B, D) float32  — L2-normalizados (||e||=1)
    labels     : (B,)   float32  — 1=anomalía, 0=normal
    tau        : temperatura de suavizado

    Returns
    -------
    Tensor escalar con el loss promedio sobre los anchors válidos.
    """
    device = embeddings.device
    labels_int = labels.long()

    pos_mask_global = (labels_int == 1)   # anomalías en el batch
    neg_mask_global = (labels_int == 0)   # normales  en el batch

    n_pos = pos_mask_global.sum().item()
    n_neg = neg_mask_global.sum().item()

    # Sin pares válidos → loss cero, sin gradiente espurio
    if n_pos < 2 or n_neg < 1:
        return torch.tensor(0.0, device=device, requires_grad=False)

    dist2 = _pairwise_squared_dist(embeddings)   # (B, B)

    anchor_indices = pos_mask_global.nonzero(as_tuple=True)[0]
    losses = []

    for i in anchor_indices:
        i = i.item()

        # Mask positivos: misma clase, excluyendo el propio anchor
        pos_mask = pos_mask_global.clone()
        pos_mask[i] = False

        if pos_mask.sum() == 0:
            continue

        # Hard Positive: anomalía más lejana
        d_ap = dist2[i][pos_mask].max()

        # Hard Negative: normal más cercana
        d_an = dist2[i][neg_mask_global].min()

        # Soft Triplet: log(1 + exp((d_ap - d_an) / τ))
        loss_i = torch.log1p(torch.exp((d_ap - d_an) / tau))
        losses.append(loss_i)

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=False)

    return torch.stack(losses).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Loop de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

bce_smooth_fn = LabelSmoothingBCE(smoothing=0.1)


def train_v2(
    loader,
    model,
    optimizer,
    device,
    triplet_weight: float = 0.3,
    tau:            float = 0.5,
):
    """
    Un epoch de entrenamiento con loss combinada BCE + Soft Triplet.

    Parameters
    ----------
    loader         : DataLoader — devuelve (features, labels)
    model          : VideoMAE_VAD_Classifier_v2
    optimizer      : torch.optim.Optimizer
    device         : torch.device
    triplet_weight : λ_t — peso del Soft Triplet en L_total
    tau            : temperatura del Soft Triplet (τ)

    Returns
    -------
    dict con losses promedio del epoch:
        total, bce, triplet
    """
    model.train()
    model = model.to(device)

    losses_total   = []
    losses_bce     = []
    losses_triplet = []
    start = time.time()

    for features, labels in loader:
        features = features.to(device, dtype=torch.float32)
        labels   = labels.to(device,   dtype=torch.float32)

        optimizer.zero_grad()

        # Forward — devuelve (score, embedding)
        scores, embeddings = model(features)
        scores = scores.float().flatten()

        # ── BCE con Label Smoothing ───────────────────────────────────────────
        loss_bce = bce_smooth_fn(scores, labels)

        # ── Soft Triplet con OHNM ─────────────────────────────────────────────
        loss_triplet = soft_triplet_ohnm(embeddings, labels, tau=tau)

        # ── Loss combinada ────────────────────────────────────────────────────
        loss = loss_bce + triplet_weight * loss_triplet

        loss.backward()
        optimizer.step()

        losses_total.append(loss.detach().cpu().item())
        losses_bce.append(loss_bce.detach().cpu().item())
        losses_triplet.append(loss_triplet.detach().cpu().item())

    elapsed = time.time() - start
    print(f"  [train] {elapsed:.1f}s | "
          f"total={np.mean(losses_total):.4f} | "
          f"bce={np.mean(losses_bce):.4f} | "
          f"triplet={np.mean(losses_triplet):.4f}")

    return {
        "total":   float(np.mean(losses_total)),
        "bce":     float(np.mean(losses_bce)),
        "triplet": float(np.mean(losses_triplet)),
    }
