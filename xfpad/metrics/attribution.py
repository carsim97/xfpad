"""Directional attribution of unseen PAIs to known training prototypes.

Implements p_{u,k} (Eq. 5 of the paper, Appendix A-A) using empirical
prototype directions: the prototype of the k-th training PAI is the unit
vector pointing from the bona fide centroid to the centroid of the k-th
PAI training cluster.

The attribution is defined identically in the 2-D X-FPAD manifold and in
the high-dimensional backbone feature space, so this module supports both
projections.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Prototypes and centroids
# ---------------------------------------------------------------------------

def compute_centroids(features: np.ndarray,
                      labels: np.ndarray,
                      class_names: Dict[int, str]) -> Dict[str, np.ndarray]:
    """Group features by integer label and return median per class."""
    out: Dict[str, np.ndarray] = {}
    for lbl, name in class_names.items():
        mask = labels == int(lbl)
        if mask.sum() == 0:
            continue
        out[name] = np.median(features[mask], axis=0)
    return out


def _prototypes_from_centroids(centroids: Dict[str, np.ndarray],
                               prototype_names: List[str],
                               bf_key: str = "Bona Fide") -> Tuple[np.ndarray, np.ndarray]:
    """Return (prototypes, bf_centroid).

    prototypes : (K, D) unit vectors, each pointing from bf_centroid to the
                 centroid of the k-th PAI training cluster.
    """
    bf_centroid = centroids[bf_key]
    rows = []
    for name in prototype_names:
        direction = centroids[name] - bf_centroid
        norm = np.linalg.norm(direction).clip(min=1e-12)
        rows.append(direction / norm)
    return np.stack(rows, axis=0), bf_centroid


# ---------------------------------------------------------------------------
# Attribution (Eq. 5)
# ---------------------------------------------------------------------------

def analyze_unseen_pais(features_train: np.ndarray,
                        labels_train: np.ndarray,
                        train_names: Dict[int, str],
                        features_unseen: np.ndarray,
                        labels_unseen: np.ndarray,
                        unseen_names: Dict[int, str],
                        prototype_order: List[str] | None = None,
                        bf_key: str = "Bona Fide",
                        tau: float = 5.0) -> Dict[str, Dict]:
    """Compute the temperature-scaled cosine softmax attribution.

    Parameters
    ----------
    features_train, labels_train, train_names :
        Training set (used to compute K prototypes via centroids).
    features_unseen, labels_unseen, unseen_names :
        Unseen-PAI validation set, projected through the same encoder /
        feature extractor.
    prototype_order :
        Optional list of class names defining the order of the K
        prototypes. Defaults to all training class names except bf_key,
        in label order.
    bf_key :
        Name of the bona fide class in train_names (default 'Bona Fide').
    tau :
        Inverse-temperature parameter (default 5.0).

    Returns
    -------
    {unseen_name: {weights, ranked_anchors, entropy}}
        weights        : (K,) attribution mass per training prototype.
        ranked_anchors : list of (name, weight) sorted descending.
        entropy        : normalised Shannon entropy in [0, 1].
    """
    centroids = compute_centroids(features_train, labels_train, train_names)
    if bf_key not in centroids:
        raise ValueError(f"bf_key '{bf_key}' not present in training centroids.")

    if prototype_order is None:
        prototype_order = [n for lbl, n in sorted(train_names.items()) if n != bf_key]

    prototypes, bf_centroid = _prototypes_from_centroids(
        centroids, prototype_order, bf_key=bf_key,
    )
    K = len(prototype_order)

    # Bucket unseen samples by class.
    unseen_buckets: Dict[str, np.ndarray] = {}
    for lbl, name in unseen_names.items():
        if name == bf_key:
            continue
        mask = labels_unseen == int(lbl)
        if mask.sum() == 0:
            continue
        unseen_buckets[name] = features_unseen[mask]

    results: Dict[str, Dict] = {}
    for pai_name, Z in unseen_buckets.items():
        Z_centered = Z - bf_centroid
        norms = np.linalg.norm(Z_centered, axis=1, keepdims=True).clip(min=1e-12)
        Z_norm = Z_centered / norms

        cos_sim = Z_norm @ prototypes.T            # (n, K)
        scaled = tau * cos_sim
        scaled -= np.max(scaled, axis=1, keepdims=True)
        exp_s = np.exp(scaled)
        sm = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        weights = np.mean(sm, axis=0)              # (K,)

        p = np.where(weights > 1e-12, weights, 1e-12)
        entropy = float(-np.sum(p * np.log(p)) / np.log(K)) if K > 1 else 0.0
        ranked = sorted(zip(prototype_order, weights.tolist()),
                        key=lambda x: x[1], reverse=True)

        results[pai_name] = {
            "weights": weights,
            "ranked_anchors": ranked,
            "entropy": entropy,
        }
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_attribution_table(results: Dict[str, Dict],
                             top_n: int | None = None) -> str:
    """Format Phase 2 results as a console-friendly table."""
    header = f"{'Unseen PAI':<35} | Top anchors (p_{{u,r}})"
    lines = [header, "-" * len(header)]
    for pai, res in results.items():
        anchors = res["ranked_anchors"]
        if top_n is not None:
            anchors = anchors[:top_n]
        anchors_str = ", ".join(f"{n} ({w:.2f})" for n, w in anchors)
        lines.append(f"{pai:<35} | {anchors_str}")
    return "\n".join(lines)
