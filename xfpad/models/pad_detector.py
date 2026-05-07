"""Unified binary PAD detector (Section IV-C of the paper).

Replaces the three near-duplicate scripts classifier_1.py, classifier_2.py,
classifier_3.py from the original codebase with a single class parameterised
by backbone name.

Convention
----------
Output is a single logit; sigmoid -> P(bona fide). This matches the original
labelling: 1 = live (bona fide), 0 = spoof. APCER and BPCER follow the same
convention (see xfpad.metrics.apcer).
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, mobilenet_v2, resnet18


# ---------------------------------------------------------------------------
# Per-backbone construction helpers
# ---------------------------------------------------------------------------

def _build_mobilenet_v2(in_channels: int) -> Tuple[nn.Module, Callable[[nn.Module, torch.Tensor], torch.Tensor]]:
    net = mobilenet_v2(weights=None)
    if in_channels != 3:
        old = net.features[0][0]
        net.features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
    net.classifier[1] = nn.Linear(1280, 1)

    def fwd(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = m.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return m.classifier(x)

    return net, fwd


def _build_resnet18(in_channels: int) -> Tuple[nn.Module, Callable[[nn.Module, torch.Tensor], torch.Tensor]]:
    net = resnet18(weights=None)
    if in_channels != 3:
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Linear(net.fc.in_features, 1)

    def fwd(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return m(x)

    return net, fwd


def _build_densenet121(in_channels: int) -> Tuple[nn.Module, Callable[[nn.Module, torch.Tensor], torch.Tensor]]:
    net = densenet121(weights=None)
    if in_channels != 3:
        old = net.features.conv0
        net.features.conv0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
    net.classifier = nn.Linear(net.classifier.in_features, 1)

    def fwd(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return m(x)

    return net, fwd


_BACKBONES = {
    "mobilenet_v2": _build_mobilenet_v2,
    "resnet18":     _build_resnet18,
    "densenet121":  _build_densenet121,
}


# ---------------------------------------------------------------------------
# Unified PAD detector
# ---------------------------------------------------------------------------

class PadDetector(nn.Module):
    """Binary PAD classifier with switchable backbone.

    Parameters
    ----------
    backbone     : 'mobilenet_v2' | 'resnet18' | 'densenet121'
    in_channels  : 1 for grayscale (default).
    """

    SUPPORTED = tuple(_BACKBONES.keys())

    def __init__(self, backbone: str = "mobilenet_v2", in_channels: int = 1) -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Supported: {sorted(_BACKBONES)}"
            )
        self.backbone_name = backbone
        self.net, self._fwd = _BACKBONES[backbone](in_channels=in_channels)
        self.net.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fwd(self.net, x)
