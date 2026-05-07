"""Geometric encoder g_psi : R^1280 -> R^2 (Section III-A).

Lightweight MLP that maps the frozen MobileNet-v2 embedding into the
two-dimensional structured latent space. Architecture matches Figure 2
of the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GeometricEncoder(nn.Module):
    """MLP geometric encoder.

    Layer plan (matches Fig. 2): LayerNorm(1280)
      -> Linear+BN+ReLU+Dropout 1280
      -> Linear+BN+ReLU+Dropout 512
      -> Linear+BN+ReLU+Dropout 512
      -> Linear+ReLU+Dropout    128
      -> Linear+ReLU+Dropout    64
      -> Linear+ReLU+Dropout    32
      -> Linear                 2

    Linear weights are initialised with orthogonal weights; the final
    projection layer is additionally scaled by 0.1 to encourage a
    slightly expanded bona fide core at the start of training.
    """

    def __init__(self,
                 input_dim: int = 1280,
                 bottleneck_dim: int = 2,
                 dropout: float = 0.2) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),

            nn.Linear(input_dim, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(32, bottleneck_dim),
        )
        self._initialize()

    def _initialize(self) -> None:
        last = self.encoder[-1]
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module is last:
                    module.weight.data.mul_(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
