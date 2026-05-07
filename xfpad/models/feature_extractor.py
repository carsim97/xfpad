"""MobileNet-v2 feature extractor f_phi (Section III-A of the paper).

Trained as a multi-class classifier; at inference time the classification
head is dropped and the 1280-D pre-classifier embedding is exposed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class FeatureExtractor(nn.Module):
    """MobileNet-v2 backbone with single-channel adaptation.

    Parameters
    ----------
    num_classes : int
        Number of training classes (1 + K). Only used in training mode.
    in_channels : int
        Input channels (1 for grayscale fingerprint patches).
    training_mode : bool
        If True, the classification head is kept and forward returns
        (logits, features). If False, forward returns only the 1280-D
        feature embedding.
    """

    EMBED_DIM = 1280

    def __init__(self,
                 num_classes: int = 8,
                 in_channels: int = 1,
                 training_mode: bool = True) -> None:
        super().__init__()
        self.training_mode = training_mode
        self.backbone = mobilenet_v2(weights=None)

        if in_channels != 3:
            old = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )

        if training_mode:
            self.backbone.classifier[1] = nn.Linear(self.EMBED_DIM, num_classes)
        else:
            self.backbone.classifier = None

        self.backbone.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self.backbone.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        features = torch.flatten(x, 1)
        if self.training_mode:
            logits = self.backbone.classifier(features)
            return logits, features
        return features
