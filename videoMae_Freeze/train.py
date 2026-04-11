"""
train.py
========
Función de entrenamiento para VideoMAE VAD Classifier.
Estructura idéntica a Step2_Detector/train.py (concatenated_train_feedback).
Lo único que cambia respecto a Step2 es el feature size:
    768 (VideoMAE ViT-Base) en lugar de 192 (X3D).
"""

import time

import numpy as np
import torch
import torch.nn as nn


loss_fn = nn.BCELoss()


def concatenated_train_feedback(loader, model, optimizer, device):
    model.train()
    model = model.to(device)

    losses = []
    start  = time.time()

    for input, labels in loader:
        input  = input.to(device).float()
        labels = labels.to(device).float()

        optimizer.zero_grad()
        logits = model(input).float().flatten()
        loss   = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().item())

    print(time.time() - start)
    return float(np.mean(losses))
