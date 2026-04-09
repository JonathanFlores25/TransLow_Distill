"""
train.py
========
Training epoch logic for VideoMAE VAD Classifier (Step3).

Design rationale vs Step2_Detector:

- No module-level side effects: Step2/train.py instantiated the device,
  parsed args, and created the loss function at import time, causing
  errors when the module was imported by mainC2FPL.py. Here all logic is
  encapsulated in train_one_epoch(); nothing executes at import.

- Criterion passed as argument (not module-level global): BCELoss with
  pos_weight is configured in mainC2FPL.py where both the weight value and
  the device are known. This avoids the init-time device/weight computation
  that broke multi-process DataLoader in Step2.

- AdamW (not SGD): the optimizer is created in mainC2FPL.py and passed
  implicitly through the training loop. The rationale for AdamW vs SGD is
  documented in option.py. train_one_epoch() is optimizer-agnostic.

- float32 throughout: explicit .float() cast on inputs and labels guards
  against accidental float64 promotion from numpy load paths.

- Returns avg_loss as Python float (not np.float32): callers can safely
  store this in a Python list and format it with standard f-strings.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import the model class with the EXACT name used in model.py and test.py.
# This is the Step3 fix: Step2 imported Model_V2 in test.py but
# Model_V3_Connection in mainC2FPL.py, making checkpoints incompatible.
from model import VideoMAE_VAD_Classifier  # noqa: F401 — referenced by callers


def train_one_epoch(
    model: VideoMAE_VAD_Classifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Run one full pass over the training DataLoader and return the mean loss.

    Parameters
    ----------
    model : VideoMAE_VAD_Classifier
        The classifier head. Expected in train() mode on entry; this function
        calls model.train() explicitly for safety in case the caller left it
        in eval() mode after a preceding validation pass.
    loader : DataLoader
        Yields (features [B, 768] float32, labels [B] float32) pairs.
        Should be constructed with BalancedBatchSampler for 50/50 balance.
    optimizer : torch.optim.Optimizer
        AdamW (or any optimizer) — called per batch: zero_grad → step.
    criterion : nn.Module
        BCELoss (with pos_weight for class imbalance). The loss function is
        passed in so mainC2FPL.py can configure pos_weight from training
        label counts at startup without requiring a global side effect here.
    device : torch.device
        Target device for inputs and labels. Model should already be on device.

    Returns
    -------
    float
        Mean batch loss over the epoch (plain Python float for JSON/CSV logging).
    """
    model.train()
    batch_losses: list[float] = []

    for features, labels in loader:
        # Move to device and ensure float32 — guards against any accidental
        # float64 promotion that can happen when collate_fn receives numpy
        # arrays with heterogeneous dtypes from different workers.
        features = features.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)

        # Standard gradient step.
        optimizer.zero_grad()

        # Forward: [B, 768] → [B, 1] probabilities ∈ [0, 1]
        scores = model(features)          # [B, 1]
        scores = scores.flatten()         # [B]

        loss = criterion(scores, labels)

        loss.backward()
        optimizer.step()

        batch_losses.append(loss.detach().cpu().item())

    if len(batch_losses) == 0:
        return 0.0

    return float(np.mean(batch_losses))
