"""Angular loss L_cos (Eq. 3) with the log-scaling penalty S(x) (Eq. 4).

K prototypes are uniformly distributed on the unit circle and rigidly
co-rotated by a single learnable scalar theta_offset. The loss is
selectively applied to attack samples (y != 0) to avoid gradient
instability at the origin.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Non-linear gradient scaling S(x), Eq. (4)
# ---------------------------------------------------------------------------

def make_log_scale(gamma_min: float = 1.0,
                   gamma_max: float = 1000.0,
                   x_min: float = 1.0,
                   x_max: float = 3.0) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return S(x) defined by Eq. (4).

    S(x) = gamma_min * 10**((x - x_min) / (x_max - x_min) * log10(gamma_max/gamma_min)) - gamma_min

    Operates on inputs already in [x_min, x_max] (the bounded range of
    1 + cosine_distance). Inputs are clamped for numerical stability.
    """
    log_ratio = math.log10(gamma_max / gamma_min)

    def s(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=x_min, max=x_max)
        exponent = (x - x_min) * log_ratio / (x_max - x_min)
        return gamma_min * torch.pow(torch.tensor(10.0, device=x.device), exponent) - gamma_min

    return s


# ---------------------------------------------------------------------------
# Angular loss
# ---------------------------------------------------------------------------

class AngularLoss(nn.Module):
    """L_cos with learnable scalar offset prototype.

    Notes
    -----
    K prototypes are placed at angles  phi_k = 2*pi*k/K + theta_offset,
    where theta_offset is the only learnable parameter. This guarantees by
    construction the uniform 2*pi/K inter-prototype separation regardless
    of optimisation, and supports incremental vocabulary extension via
    update_K (used in Phase 3 ablations).

    Label convention
    ----------------
    PAI labels must be contiguous integers in [1, K]. Label 0 is bona
    fide and contributes no gradient to L_cos.
    """

    def __init__(self, log_scale_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.cosine = nn.CosineEmbeddingLoss(reduction="none")
        self.log_scale = log_scale_fn
        self.K: int | None = None
        self.theta_offset: nn.Parameter | None = None

    # ---- prototype management ------------------------------------------------

    def _init_offset(self, K: int, device: torch.device | str) -> None:
        self.K = int(K)
        self.theta_offset = nn.Parameter(torch.zeros(1, device=device))

    def update_K(self, new_K: int, device: torch.device | str) -> None:
        """Re-initialise theta_offset for a different K (vocabulary change)."""
        self._init_offset(new_K, device)

    def normalized_prototypes(self) -> torch.Tensor:
        if self.K is None or self.theta_offset is None:
            raise RuntimeError("AngularLoss is not initialised; call update_K first.")
        device = self.theta_offset.device
        angles = torch.tensor(
            [2 * math.pi * k / self.K for k in range(self.K)],
            device=device,
        ) + self.theta_offset
        return torch.stack([angles.cos(), angles.sin()], dim=1)

    # ---- forward -------------------------------------------------------------

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.K is None:
            # Auto-init from the first batch's labels.
            pai_labels = labels[labels != 0].unique()
            expected = torch.arange(1, len(pai_labels) + 1, device=labels.device)
            if not torch.equal(pai_labels, expected):
                raise ValueError(
                    "PAI labels must be contiguous integers in [1, K]; "
                    f"got {pai_labels.tolist()}."
                )
            self._init_offset(len(pai_labels), z.device)

        max_label = int(labels[labels != 0].max().item()) if (labels != 0).any() else 0
        if max_label > self.K:
            raise ValueError(
                f"PAI label {max_label} exceeds K={self.K}. "
                "Call update_K before fine-tuning with new PAIs."
            )

        mask = labels != 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)

        z_sel = z[mask]
        labels_sel = labels[mask]
        protos = self.normalized_prototypes()
        targets = protos[labels_sel - 1]

        cos_dist = self.cosine(
            z_sel,
            targets,
            torch.ones(z_sel.size(0), device=z.device),
        )
        return self.log_scale(1.0 + cos_dist).mean()
