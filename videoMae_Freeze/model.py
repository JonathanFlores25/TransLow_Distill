"""
model.py
========
VideoMAE VAD Classifier — misma arquitectura que Model_V3_Connection
(Step2_Detector) adaptada para features de 768 dims (VideoMAE ViT-Base)
en lugar de 192 dims (X3D).

Estructura:
    768 → fc1(768→512) + fc_att1(768→512,Softmax) residual → ReLU → Dropout(0.6)
    512 → fc2(512→32)  + fc_att2(512→32, Softmax) residual → ReLU → Dropout(0.6)
    32  → fc3(32→1) → Sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.init as torch_init


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class VideoMAE_VAD_Classifier(nn.Module):
    """
    Clasificador binario de anomalías sobre features VideoMAE (768,).

    Idéntico a Model_V3_Connection de Step2_Detector salvo por
    feature_dim=768 (VideoMAE ViT-Base) vs 192 (X3D).

    Input  : [B, 768]  float32
    Output : [B, 1]    float32  — probabilidad de anomalía ∈ [0, 1]
    """

    def __init__(self, feature_dim: int = 768) -> None:
        super().__init__()

        self.fc1     = nn.Linear(feature_dim, 512)
        self.fc_att1 = nn.Sequential(nn.Linear(feature_dim, 512), nn.Softmax(dim=1))

        self.fc2     = nn.Linear(512, 32)
        self.fc_att2 = nn.Sequential(nn.Linear(512, 32), nn.Softmax(dim=1))

        self.fc3     = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.6)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.apply(weight_init)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Primera capa + atención
        x    = self.fc1(inputs)
        att1 = self.fc_att1(inputs)
        x    = (x * att1) + x      # residual
        x    = self.relu(x)
        x    = self.dropout(x)

        # Segunda capa + atención
        att2 = self.fc_att2(x)
        x    = self.fc2(x)
        x    = (x * att2) + x      # residual
        x    = self.relu(x)
        x    = self.dropout(x)

        # Salida
        x = self.sigmoid(self.fc3(x))
        return x
