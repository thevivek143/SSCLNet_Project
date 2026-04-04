# FILE: models/encoder.py

import torch.nn as nn
import timm


class DenseNetEncoder(nn.Module):
    """
    DenseNet121 encoder with global average pooling.
    Designed for controlled fine-tuning in medical imaging tasks.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            model_name="densenet121",
            pretrained=pretrained,
            num_classes=0,          # remove classifier
            global_pool="avg"       # (B, 1024)
        )

        self.out_features = 1024

    def forward(self, x):
        return self.backbone(x)

    def freeze_all(self):
        """Freeze entire encoder"""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_block(self):
        """
        Unfreeze DenseNet final block for fine-tuning.
        Safe + effective for limited medical datasets.
        """
        for name, p in self.backbone.named_parameters():
            if "denseblock4" in name or "norm5" in name:
                p.requires_grad = True
