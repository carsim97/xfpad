"""Concentric radial loss (Eq. 1 of the paper).

Bona fide samples (label 0) are pushed inside a disc of radius rho_bf;
PAI samples (label > 0) are pushed beyond rho_pa = rho_bf + delta_rho.
The decision threshold is T = rho_bf^2.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConcentricLoss(nn.Module):
    """L_conc as defined in Eq. (1) of the paper.

    Parameters
    ----------
    rho_bf      : bona fide radius. Threshold T = rho_bf^2.
    delta_rho   : radial margin. PAI radius rho_pa = rho_bf + delta_rho.
    """

    def __init__(self, rho_bf: float = 1.0, delta_rho: float = 1.0) -> None:
        super().__init__()
        self.rho_bf = float(rho_bf)
        self.delta_rho = float(delta_rho)
        self.rho_pa = self.rho_bf + self.delta_rho

    @property
    def threshold(self) -> float:
        """Squared decision threshold T = rho_bf^2."""
        return self.rho_bf ** 2

    def update_radius(self, new_rho_bf: float) -> None:
        """Adjust rho_bf at runtime; rho_pa is rebuilt to preserve the margin."""
        self.rho_bf = float(new_rho_bf)
        self.rho_pa = self.rho_bf + self.delta_rho

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(z, dim=1)
        zero = torch.zeros_like(norms)
        loss_bf = torch.where(
            (labels == 0) & (norms > self.rho_bf),
            (norms - self.rho_bf) ** 2,
            zero,
        )
        loss_pa = torch.where(
            (labels != 0) & (norms < self.rho_pa),
            (self.rho_pa - norms) ** 2,
            zero,
        )
        return (loss_bf + loss_pa).mean()
