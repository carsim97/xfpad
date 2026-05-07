"""Geometric diagnostic metrics: BFO, RCI, ACS (Appendix A-B of the paper).

Definitions follow Eqs. (6)-(8) and operate on the trained 2-D manifold
projections of the training set.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def calculate_metrics(latent_vectors: np.ndarray,
                      labels: np.ndarray,
                      rho_bf: float = 1.0) -> Tuple[float, float, float]:
    """Compute (BFO, RCI, ACS) on the projected training set.

    Parameters
    ----------
    latent_vectors : (N, 2) array of g_psi(z) projections.
    labels         : (N,) integer labels (0 for bona fide, > 0 for PAIs).
    rho_bf         : bona fide radius hyperparameter (= sqrt(T)).

    Returns
    -------
    (bfo, rci, acs) all in [0, 1]. Returns (0, 0, 0) if either bona
    fide or PAI samples are absent.
    """
    z = np.asarray(latent_vectors)
    y = np.asarray(labels)

    mask_bf = (y == 0)
    mask_pai = (y > 0)
    if not np.any(mask_bf) or not np.any(mask_pai):
        return 0.0, 0.0, 0.0

    mags_bf = np.linalg.norm(z[mask_bf], axis=1)
    mags_pai = np.linalg.norm(z[mask_pai], axis=1)

    # Global manifold scale (median PAI radial magnitude).
    global_scale = np.median(mags_pai) + 1e-8

    # 1) BFO - Bona Fide Occupancy (Eq. 6).
    core_radius = max(np.percentile(mags_bf, 95), rho_bf)
    bfo = float(np.clip(core_radius / global_scale, 0.0, 1.0))

    # Per-class statistics over the K PAI classes.
    unique_pai = np.unique(y[mask_pai])
    K = len(unique_pai)

    p5_per_class = []
    acs_per_class = []
    theta_limit = np.pi / K  # half the angular spacing between adjacent prototypes

    for cls in unique_pai:
        z_cls = z[y == cls]
        mags_cls = np.linalg.norm(z_cls, axis=1)

        # 5th percentile radial magnitude (frontline).
        p5_per_class.append(np.percentile(mags_cls, 5))

        # Mean Resultant Length R_k.
        unit = z_cls / (mags_cls[:, None] + 1e-8)
        R_k = float(np.linalg.norm(np.mean(unit, axis=0)))
        R_k = float(np.clip(R_k, 0.0, 1.0))

        # Angular deviation s = sqrt(2 * (1 - R)).
        s = float(np.sqrt(2.0 * (1.0 - R_k)))

        score = max(0.0, 1.0 - s / theta_limit)
        acs_per_class.append(score)

    # 2) RCI - Radial Clearance Index (Eq. 7).
    characteristic_frontline = float(np.median(p5_per_class))
    rci = max(0.0, (characteristic_frontline - rho_bf) / global_scale)

    # 3) ACS - Angular Cohesion Score (Eq. 8).
    acs = float(np.mean(acs_per_class))

    return bfo, rci, acs
