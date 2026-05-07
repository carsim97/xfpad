"""Substring-based label assignment matching the original X-FPAD code.

A label rule is a tuple of substrings; a path receives a given label if all
substrings appear in it. The first matching rule wins. This preserves the
behaviour of the original `build_labels` in geometric.py.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def build_labels(image_paths: Sequence[str],
                 label_map: Dict[Tuple[str, ...], int]) -> List[int]:
    """Assign one integer label per path.

    Parameters
    ----------
    image_paths : list of paths.
    label_map   : {(substring1, substring2, ...): label}; rules tested in order.

    Raises
    ------
    ValueError : if a path matches no rule.
    """
    labels: List[int] = []
    for path in image_paths:
        for substrs, lbl in label_map.items():
            if all(s in path for s in substrs):
                labels.append(int(lbl))
                break
        else:
            raise ValueError(f"No label rule matches: {path}")
    return labels


def assert_contiguous_pai_labels(labels: Sequence[int]) -> int:
    """Verify that PAI labels (label > 0) form a contiguous range [1, K] and return K."""
    arr = np.asarray(list(labels))
    pai = np.unique(arr[arr > 0])
    if pai.size == 0:
        raise ValueError("No PAI labels found (all zero).")
    expected = np.arange(1, len(pai) + 1)
    if not np.array_equal(pai, expected):
        raise ValueError(
            f"PAI labels must be contiguous integers in [1, K]. "
            f"Got: {pai.tolist()}, expected: {expected.tolist()}"
        )
    return int(pai.max())
