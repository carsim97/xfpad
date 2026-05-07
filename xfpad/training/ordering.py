"""Circular ordering of PAI prototypes around the manifold.

Given the K x K cosine-similarity matrix between PAI centroids, we look
for the cyclic ordering that maximises the sum of similarities between
adjacent prototypes (i.e. neighbouring angular sectors). For K <= 8 we
exhaustively search the (K-1)!/2 distinct circular permutations; for
K > 8 we fall back to nearest-neighbour seed + 2-opt local search.

The ordering is purely a presentational aid: it determines how training
labels are mapped to the K angular sectors, so that materials with
similar feature signatures end up in adjacent sectors of the manifold.
"""
from __future__ import annotations

import itertools
import sys
from typing import List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Centroids and similarity
# ---------------------------------------------------------------------------

def compute_pai_centroids(features: np.ndarray,
                          labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean feature vector per PAI class.

    PAI labels must form a contiguous range [1, K]. Returns
    (centroids of shape (K, D), pai_labels of shape (K,)).
    """
    pai_mask = labels > 0
    if not pai_mask.any():
        raise ValueError("No PAI samples found (all labels are 0).")

    pai_labels = np.unique(labels[pai_mask])
    expected = np.arange(1, len(pai_labels) + 1)
    if not np.array_equal(pai_labels, expected):
        raise ValueError(
            f"PAI labels must be contiguous integers in [1, K]. "
            f"Got: {pai_labels.tolist()}, expected: {expected.tolist()}"
        )

    centroids = np.stack(
        [features[labels == lbl].mean(axis=0) for lbl in pai_labels],
        axis=0,
    )
    return centroids, pai_labels


def cosine_similarity_matrix(centroids: np.ndarray) -> np.ndarray:
    """K x K cosine similarity matrix between row vectors."""
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    normed = centroids / np.clip(norms, 1e-12, None)
    return normed @ normed.T


# ---------------------------------------------------------------------------
# Cycle scoring and equivalence
# ---------------------------------------------------------------------------

def cumulative_circular_similarity(order: Sequence[int],
                                   S: np.ndarray) -> float:
    """Sum_{k=0..K-1} S[order[k], order[(k+1) % K]]."""
    K = len(order)
    return float(sum(S[order[k], order[(k + 1) % K]] for k in range(K)))


def canonical_circular_form(order: Sequence[int]) -> Tuple[int, ...]:
    """Lexicographically smallest representative under rotation+reflection."""
    K = len(order)
    candidates = []
    for shift in range(K):
        rotated = tuple(order[(i + shift) % K] for i in range(K))
        candidates.append(rotated)
        candidates.append(tuple(reversed(rotated)))
    return min(candidates)


# ---------------------------------------------------------------------------
# Search algorithms
# ---------------------------------------------------------------------------

def _exhaustive_search(S: np.ndarray) -> Tuple[List[int], float]:
    K = S.shape[0]
    best_score = -np.inf
    best_order: Tuple[int, ...] | None = None
    remaining = list(range(1, K))
    for perm in itertools.permutations(remaining):
        # Factor out reflections.
        if perm[0] > perm[-1]:
            continue
        order = (0,) + perm
        score = cumulative_circular_similarity(order, S)
        if score > best_score:
            best_score = score
            best_order = order
    assert best_order is not None
    return list(best_order), float(best_score)


def _greedy_seed(S: np.ndarray) -> List[int]:
    K = S.shape[0]
    visited = [False] * K
    order = [0]
    visited[0] = True
    while len(order) < K:
        current = order[-1]
        best_next = -1
        best_sim = -np.inf
        for j in range(K):
            if not visited[j] and S[current, j] > best_sim:
                best_sim = S[current, j]
                best_next = j
        order.append(best_next)
        visited[best_next] = True
    return order


def _two_opt(order: List[int], S: np.ndarray, max_iter: int = 10000) -> Tuple[List[int], float]:
    K = len(order)
    order = list(order)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        cur_score = cumulative_circular_similarity(order, S)
        for i in range(K - 1):
            for j in range(i + 1, K):
                new_order = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                new_score = cumulative_circular_similarity(new_order, S)
                if new_score > cur_score:
                    order = new_order
                    cur_score = new_score
                    improved = True
    return order, float(cur_score)


def find_optimal_circular_ordering(S: np.ndarray) -> Tuple[List[int], float, str]:
    """Return (order, score, method).

    Exhaustive search for K <= 8, greedy + 2-opt for K > 8.
    """
    K = S.shape[0]
    if K <= 8:
        order, score = _exhaustive_search(S)
        return order, score, "exhaustive"
    print(f"[ordering] K = {K} > 8, falling back to greedy + 2-opt.", file=sys.stderr)
    seed = _greedy_seed(S)
    order, score = _two_opt(seed, S)
    return order, score, "greedy+2opt"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_ordering_report(order: Sequence[int],
                           S: np.ndarray,
                           current_labels: Sequence[int],
                           class_names: Sequence[str],
                           score: float | None = None,
                           method: str | None = None) -> str:
    """Build a multi-line console report describing the chosen ordering."""
    K = len(order)
    width = max(max(len(n) for n in class_names), 8)
    out: List[str] = []
    out.append("=" * 60)
    out.append("OPTIMAL CIRCULAR ORDERING")
    out.append("=" * 60)
    if method:
        out.append(f"Method     : {method}")
    if score is not None:
        out.append(f"Total cum. similarity : {score:.4f}")
        out.append(f"Average per-edge      : {score / K:.4f}")
    out.append("")

    out.append("Order around the circle:")
    out.append(f"  {'Pos':>3}  {'New label':>10}  "
               f"{'Material':<{width}}  {'Old label':>10}")
    for new_pos, idx in enumerate(order):
        new_label = new_pos + 1
        old_label = int(current_labels[idx])
        out.append(f"  {new_pos:>3}  {new_label:>10}  "
                   f"{class_names[idx]:<{width}}  {old_label:>10}")

    out.append("")
    out.append("Adjacency similarities along the cycle:")
    for k in range(K):
        a = order[k]
        b = order[(k + 1) % K]
        out.append(f"  {class_names[a]:<{width}} -- {class_names[b]:<{width}} : "
                   f"{S[a, b]:+.3f}")
    return "\n".join(out)
