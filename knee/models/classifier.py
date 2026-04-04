# FILE: models/classifier.py

import torch.nn as nn
from models.attention import SEBlock


class KneeClassifier(nn.Module):
    """
    Knee OA classifier with:
    - CNN encoder
    - SE attention
    - Regularized classification head
    """
    def __init__(self, encoder, num_classes: int = 3):
        super().__init__()

        self.encoder = encoder
        feat_dim = encoder.out_features

        self.attention = SEBlock(feat_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x
