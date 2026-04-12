"""
model_v2.py
===========
VideoMAE VAD Classifier v2 — Metric Learning Edition.

Cambios respecto a v1 (VideoMAE_VAD_Classifier):
-------------------------------------------------
  · Encoder compartido: fc1+att1 (768→512) recibe gradiente de AMBOS losses.
  · Classification Head: fc2+att2+fc3 (512→32→1), solo recibe ∇BCE.
  · Projection Head: MLP (512→256→BN→ReLU→128) + F.normalize, solo ∇Triplet.
  · forward() devuelve (score, embedding) durante entrenamiento.
  · predict() devuelve solo score — usado por test.py sin cambios.

Flujo de gradientes
-------------------
  L_total = L_bce_smooth + λ_t · L_soft_triplet
                │                      │
                ▼                      ▼
        ClassificationHead       ProjectionHead
                │                      │
                └──────────┬───────────┘
                           ▼
                    SharedEncoder (fc1+att1)
                    ← aprende de ambos losses

Compatibilidad futura
---------------------
  Fase 2: descongelar última capa VideoMAE + conv temporal →
          ProjectionHead se reconecta al nuevo output; el resto no cambia.
  Fase 3: loss de atención guiada se añade sobre la estructura existente.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init


# ─────────────────────────────────────────────────────────────────────────────
# Inicialización de pesos
# ─────────────────────────────────────────────────────────────────────────────

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


# ─────────────────────────────────────────────────────────────────────────────
# Projection Head
# ─────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    MLP de proyección: 512 → 256 → 128 (normalizado en la hiperesfera unitaria).

    Diseño: Linear → BatchNorm1d → ReLU → Linear → F.normalize
    El BatchNorm estabiliza el entrenamiento con pseudo-labels ruidosas.
    La normalización L2 final coloca los embeddings en S^127 (unit hypersphere),
    lo que acota las distancias en [0, 2] y hace el Soft Triplet más estable.

    Input  : [B, 512]  float32   (salida del SharedEncoder)
    Output : [B, 128]  float32   embeddings L2-normalizados
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=1)   # ||z||_2 = 1


# ─────────────────────────────────────────────────────────────────────────────
# Modelo principal v2
# ─────────────────────────────────────────────────────────────────────────────

class VideoMAE_VAD_Classifier_v2(nn.Module):
    """
    Clasificador binario de anomalías con Metric Learning.

    Arquitectura
    ------------
    Input (768,)
        │
    [SharedEncoder]  fc1(768→512) + fc_att1(Softmax) residual → ReLU → Dropout(0.6)
        │
        ├── [ClassificationHead]  fc2(512→32)+att2 → ReLU → Dropout → fc3(32→1) → Sigmoid
        │       → score  (1,)   para BCE Loss
        │
        └── [ProjectionHead]  Linear(512→256) → BN → ReLU → Linear(256→128) → F.normalize
                → embedding (128,)  para Soft Triplet Loss

    Uso en entrenamiento
    --------------------
        score, emb = model(features)          # devuelve ambas salidas
        loss = bce(score, labels) + λ * soft_triplet(emb, labels)

    Uso en evaluación (test.py sin cambios)
    ----------------------------------------
        score = model.predict(features)       # solo el score
    """

    def __init__(
        self,
        feature_dim: int = 768,
        proj_hidden:  int = 256,
        proj_out:     int = 128,
    ) -> None:
        super().__init__()

        # ── Encoder compartido (768 → 512) ───────────────────────────────────
        self.fc1     = nn.Linear(feature_dim, 512)
        self.fc_att1 = nn.Sequential(nn.Linear(feature_dim, 512), nn.Softmax(dim=1))

        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.6)

        # ── Classification Head (512 → 32 → 1) ───────────────────────────────
        self.fc2     = nn.Linear(512, 32)
        self.fc_att2 = nn.Sequential(nn.Linear(512, 32), nn.Softmax(dim=1))
        self.fc3     = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        # ── Projection Head (512 → 256 → 128, normalizado) ───────────────────
        self.proj_head = ProjectionHead(
            in_dim=512, hidden_dim=proj_hidden, out_dim=proj_out
        )

        self.apply(weight_init)

    # ── Forward interno ───────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder compartido: (B, 768) → (B, 512).

        SIN dropout aquí: tanto ClassificationHead como ProjectionHead
        reciben el mismo feature limpio. El dropout en la rama de proyección
        destruiría el aprendizaje de distancias (metric collapse a ln(2)).
        El dropout se aplica solo dentro de _classify.
        """
        feat = self.fc1(x)
        att  = self.fc_att1(x)
        feat = (feat * att) + feat      # residual con atención
        feat = self.relu(feat)
        return feat

    def _classify(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Classification Head: (B, 512) → (B, 1).
        Dropout aplicado aquí — no afecta la rama de proyección.
        """
        feat = self.dropout(feat)       # dropout solo para la clasificación
        att2 = self.fc_att2(feat)
        x    = self.fc2(feat)
        x    = (x * att2) + x          # residual con atención
        x    = self.relu(x)
        x    = self.dropout(x)
        return self.sigmoid(self.fc3(x))

    # ── API pública ───────────────────────────────────────────────────────────

    def forward(self, inputs: torch.Tensor):
        """
        Entrenamiento: devuelve (score, embedding).

        score     : (B, 1)    probabilidad de anomalía ∈ [0,1]  → BCE Loss
        embedding : (B, 128)  vector L2-normalizado              → Soft Triplet Loss
        """
        feat = self._encode(inputs)
        score = self._classify(feat)
        emb   = self.proj_head(feat)
        return score, emb

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Evaluación: devuelve solo el score (B, 1).
        Compatible con test_Over() y test_Over_online() de v1 sin cambios.
        """
        with torch.no_grad():
            feat  = self._encode(inputs)
            score = self._classify(feat)
        return score
