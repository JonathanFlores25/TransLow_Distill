"""
model.py
========
VideoMAE VAD Classifier architecture (Step3).

Design rationale vs Step2_Detector:

1. LayerNorm as first operation (not BatchNorm):
   VideoMAE patch tokens are mean-pooled over 1568 spatiotemporally entangled
   positions. The resulting 768-dim vector does NOT have i.i.d. dimensions —
   dimensions encode joint temporal×spatial structure due to the tube embedding
   (tubelet_size=2, patch_size=16). BatchNorm normalizes across the batch
   dimension and is designed for feature maps where each channel IS
   semantically independent. LayerNorm normalizes each sample independently
   across its 768 dims, which is the correct inductive bias here: it respects
   that a single (768,) vector is a holistic spatiotemporal encoding.

2. GELU activation (not ReLU):
   VideoMAE itself uses GELU in all FFN layers (consistent with ViT/BERT
   family). Using GELU in the downstream classifier keeps the activation
   landscape smooth and consistent with the upstream representation, which
   was optimized assuming GELU-friendly gradient flow.

3. Soft-attention gating branch on the original 768-dim input:
   The 768 dims are spatiotemporally entangled — some dimensions encode
   temporal dynamics (motion, action transitions) and others encode spatial
   semantics (object appearance). For anomaly detection, we want the model
   to selectively amplify dimensions that are discriminative for surveillance
   anomalies vs. Kinetics clean actions. The sigmoid gate learns a soft
   importance mask over the 768 dims: gate = sigmoid(Linear(768→768)).
   The output gate*x + x is a residual gating that preserves the original
   feature while allowing per-dimension scaling. This is applied to the
   original input BEFORE projection to 512, so the 512-dim representation
   already carries the domain-adapted weighting.
   NOTE: gate is applied to the original 768 input (before projection) and
   the gated output is added back to the original — this forms a residual
   skip that prevents the gate from completely zeroing informative dims.

4. Dropout schedule: 0.3 after first projection, 0.2 after second.
   Higher dropout early in the network compensates for domain shift between
   Kinetics pre-training and UCF-Crime. Lower dropout later preserves the
   learned discriminative representation. Step2 used 0.6 throughout, which
   is aggressive and suitable for raw pixel features but over-regularizes
   compact transformer embeddings.

5. Single class output with Sigmoid (binary anomaly score in [0, 1]):
   Each clip produces one anomaly probability. At test time these are
   repeated × clip_step and compared against frame-level binary GT.

IMPORTANT: the class name VideoMAE_VAD_Classifier must be identical in
model.py, train.py, and test.py. Do not alias or rename.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# The only supported input dimension — fixed by VideoMAE ViT-Base architecture.
# Asserting this in forward() catches dimension mismatches from wrong backbone
# features before they propagate to NaN losses.
_EXPECTED_FEATURE_DIM = 768


class VideoMAE_VAD_Classifier(nn.Module):
    """
    Lightweight anomaly head for VideoMAE clip features.

    Input  : [B, 768]  float32  — mean-pooled spatiotemporal tube embeddings
    Output : [B, 1]    float32  — anomaly probability in [0, 1]

    Architecture overview::

        x (768,)
          │
          ├─► LayerNorm(768)                        # stabilise domain-shifted features
          │       │
          │       ├─► fc1: Linear(768→512) → GELU → Dropout(0.3)   # main branch
          │       │
          │       └─► gate branch (on original x before LayerNorm):
          │              gate = Sigmoid(Linear(768→768))
          │              gated = gate * x_orig + x_orig              # residual gating
          │
          │   [gated is added to the 512-dim projection output
          │    via a learned mixing linear: Linear(768→512)]
          │
          ├─► Linear(512→128) → GELU → Dropout(0.2)
          │
          └─► Linear(128→1) → Sigmoid

    The soft-attention branch operates on the RAW input (before LayerNorm)
    because the gate is learning which of the 768 spatiotemporal dimensions
    are relevant for surveillance anomaly detection — it should see the
    original Kinetics-domain feature magnitudes to form its weighting.
    """

    def __init__(self, feature_dim: int = _EXPECTED_FEATURE_DIM) -> None:
        super().__init__()

        if feature_dim != _EXPECTED_FEATURE_DIM:
            raise ValueError(
                f"VideoMAE_VAD_Classifier only supports feature_dim={_EXPECTED_FEATURE_DIM} "
                f"(VideoMAE ViT-Base hidden_size). Got feature_dim={feature_dim}. "
                f"Re-extract features with a different backbone to use a different dim."
            )

        self.feature_dim = feature_dim

        # ── Normalisation ────────────────────────────────────────────────────
        # LayerNorm across the 768-dim feature vector (per-sample, not
        # per-batch). Reduces the effect of the Kinetics→UCF-Crime domain
        # shift by re-centering each sample's feature distribution.
        self.layer_norm = nn.LayerNorm(feature_dim)

        # ── Main projection branch ───────────────────────────────────────────
        # Linear(768→512) + GELU + Dropout(0.3): compresses the spatiotemporal
        # representation into a discriminative anomaly-oriented subspace.
        # GELU (Gaussian Error Linear Unit) is used throughout to match the
        # activation function in VideoMAE's own FFN layers.
        self.fc1 = nn.Linear(feature_dim, 512)
        self.drop1 = nn.Dropout(p=0.3)

        # ── Soft-attention gating branch ─────────────────────────────────────
        # Applied to the ORIGINAL (un-normed) 768-dim input.
        # gate = Sigmoid(Linear(768→768)) — learns per-dimension importance.
        # output = gate * x + x — residual gating: can suppress irrelevant dims
        # (e.g., scene-type channels that fire for Kinetics actions but not for
        # surveillance anomalies) while preserving the rest via the skip.
        # A second linear then mixes the gated 768-dim signal into 512-dim
        # space to be added to the main branch before the second projection.
        self.gate_linear = nn.Linear(feature_dim, feature_dim)
        # Projects gated signal into the same 512-dim space as the main branch
        # so they can be summed before passing to fc2.
        self.gate_proj = nn.Linear(feature_dim, 512)

        # ── Second projection ────────────────────────────────────────────────
        # 512→128: further compression with lighter regularisation (0.2) since
        # the representation has already been regularised by Dropout(0.3).
        self.fc2 = nn.Linear(512, 128)
        self.drop2 = nn.Dropout(p=0.2)

        # ── Output head ──────────────────────────────────────────────────────
        # Single sigmoid output: anomaly probability ∈ [0, 1].
        # Compatible with BCELoss(pos_weight=...) in train.py.
        self.fc3 = nn.Linear(128, 1)

        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        # Initialise weights with Xavier uniform for linear layers and zeros
        # for biases — a neutral starting point that does not bias the gate
        # toward full suppression or full pass-through at epoch 0.
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm starts as identity (scale=1, bias=0) — standard.
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape [B, 768], dtype float32
            Mean-pooled VideoMAE patch token embeddings from FE_VideoMAE.py.

        Returns
        -------
        torch.Tensor, shape [B, 1], dtype float32
            Anomaly probability ∈ [0, 1] for each clip.
        """
        # Hard contract: input must be exactly 768-dim VideoMAE features.
        # Asserts here rather than in __init__ so shape errors surface at
        # forward time with batch context (easier debugging).
        assert x.shape[-1] == _EXPECTED_FEATURE_DIM, (
            f"VideoMAE_VAD_Classifier.forward() expected last dim={_EXPECTED_FEATURE_DIM} "
            f"but got shape {tuple(x.shape)}. "
            f"Ensure features were extracted with FE_VideoMAE.py "
            f"(MCG-NJU/videomae-base-finetuned-kinetics, hidden_size=768)."
        )

        # Keep a reference to the raw input for the gating branch.
        # The gate learns from the Kinetics-domain magnitudes (before LN)
        # so it can selectively amplify surveillance-relevant dimensions.
        x_orig = x  # [B, 768]

        # ── Soft-attention gating (on raw input) ─────────────────────────────
        # gate ∈ (0,1)^768 — per-dimension attention mask.
        # Residual form (gate*x + x) ensures gradient flows through the skip
        # even if gate saturates toward 0 early in training.
        gate = self.sigmoid(self.gate_linear(x_orig))   # [B, 768]
        x_gated = gate * x_orig + x_orig                 # [B, 768] residual gated

        # Project gated signal to 512-dim to match main branch dimensionality.
        gated_proj = self.gelu(self.gate_proj(x_gated))  # [B, 512]

        # ── Main branch ───────────────────────────────────────────────────────
        x_norm = self.layer_norm(x_orig)                  # [B, 768] layer-normed
        x_main = self.gelu(self.fc1(x_norm))              # [B, 512]
        x_main = self.drop1(x_main)                       # [B, 512] dropout(0.3)

        # ── Merge branches ────────────────────────────────────────────────────
        # Adding the gated projection to the main branch allows the model to
        # selectively re-weight dimensions suppressed by LayerNorm when the
        # gate determines they carry anomaly-relevant magnitude information.
        x_merged = x_main + gated_proj                    # [B, 512]

        # ── Second projection ─────────────────────────────────────────────────
        x_out = self.gelu(self.fc2(x_merged))             # [B, 128]
        x_out = self.drop2(x_out)                         # [B, 128] dropout(0.2)

        # ── Output ────────────────────────────────────────────────────────────
        out = self.sigmoid(self.fc3(x_out))               # [B, 1] ∈ [0,1]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# DEPRECATED variants (kept for reference, do not use in new code)
# ─────────────────────────────────────────────────────────────────────────────

# DEPRECATED: Model — used in Step2_Detector
# Bugs: _init_ instead of __init__ (typo), ReLU instead of GELU,
# no LayerNorm, no attention branch, hardcoded feature_dim.
# class Model(nn.Module):
#     def _init_(self, n_features):  # BUG: single underscore
#         super(Model, self)._init_()
#         self.fc1 = nn.Linear(n_features, 512)
#         self.fc2 = nn.Linear(512, 32)
#         self.fc3 = nn.Linear(32, 1)
#         self.dropout = nn.Dropout(0.6)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

# DEPRECATED: Model_V2 — used in Step2_Detector test.py
# Improvement over Model (proper __init__) but still uses ReLU, no LN,
# no attention, 0.6 dropout throughout, generic n_features arg.
# class Model_V2(nn.Module):
#     def __init__(self, n_features):
#         ...

# DEPRECATED: Model_V3_Connection — used in Step2_Detector mainC2FPL.py
# Added attention with Softmax (not Sigmoid), ReLU, no LN.
# Softmax attention over 512 dims creates a normalised competition where
# increasing one attention weight must decrease others — inappropriate for
# anomaly detection where multiple dimensions can simultaneously be salient.
# Replaced by VideoMAE_VAD_Classifier which uses per-dimension sigmoid gates.
# class Model_V3_Connection(nn.Module):
#     def __init__(self, n_features):
#         ...
