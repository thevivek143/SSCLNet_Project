# FILE: models/attention.py

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block adapted for
    feature vectors (B, C) instead of feature maps.

    This improves channel-wise feature reweighting
    for medical imaging classification.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: Tensor of shape (B, C)
        """
        scale = self.fc(x)
        return x * scale
