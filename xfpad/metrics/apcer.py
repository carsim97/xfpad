"""APCER, BPCER, and per-unseen-PAI APCER computation.

Convention (preserved from the original codebase)
-------------------------------------------------
- Predictions are sigmoid logits thresholded at 0.5; output = 1 means
  "live / bona fide", output = 0 means "spoof".
- Ground truth: 1 = live (bona fide), 0 = spoof (any PAI).
- BPCER = fraction of bona fide samples classified as spoof.
- APCER = fraction of spoof samples classified as bona fide.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def apcer_bpcer(y_true: Sequence[int],
                y_pred: Sequence[int]) -> Tuple[float, float, float]:
    """Return (APCER, BPCER, ACE) as percentages.

    ACE = (APCER + BPCER) / 2 is the Average Classification Error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_live = int(np.sum(y_true == 1))
    n_spoof = int(np.sum(y_true == 0))
    if n_live == 0 or n_spoof == 0:
        raise ValueError("Both bona fide and spoof samples must be present.")

    bpcer = float(np.sum((y_true == 1) & (y_pred == 0)) / n_live * 100.0)
    apcer = float(np.sum((y_true == 0) & (y_pred == 1)) / n_spoof * 100.0)
    ace = (apcer + bpcer) / 2.0
    return apcer, bpcer, ace


def apcer_per_unseen_pai(image_paths: Sequence[str],
                         y_pred: Sequence[int],
                         pai_definitions: Dict[str, Sequence[str]]) -> Dict[str, float]:
    """Per-PAI APCER computed by substring matching on image paths.

    Parameters
    ----------
    image_paths     : ordered file paths corresponding to each prediction.
    y_pred          : binary predictions aligned to image_paths.
    pai_definitions : {pai_name: list of substrings (lowercased) that must
                      ALL appear in the path for a sample to belong to that
                      PAI category}. Matching is case-insensitive.

    Returns
    -------
    {pai_name: APCER (%)}.
    """
    paths_lc = [p.lower() for p in image_paths]
    y_pred = np.asarray(y_pred)

    out: Dict[str, float] = {}
    for pai_name, substrs in pai_definitions.items():
        substrs_lc = [s.lower() for s in substrs]
        mask = np.array([all(s in p for s in substrs_lc) for p in paths_lc])
        n = int(mask.sum())
        if n == 0:
            out[pai_name] = float("nan")
            continue
        misclassified = int(np.sum(mask & (y_pred == 1)))
        out[pai_name] = misclassified / n * 100.0
    return out
