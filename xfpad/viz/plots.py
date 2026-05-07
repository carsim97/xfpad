"""2-D latent space plotting (Figures 4-6 in the paper).

Two modes:
    - 'training' : per-class scatter, all classes coloured, legend.
    - 'unseen'   : a separate figure per unseen PAI, all unseen samples in
                   one colour, bona fide overlay optional.

The dashed bona fide circle of radius sqrt(T) is drawn when T is provided.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _draw_threshold_circle(ax, T: float, color: str = "#3a6fb5") -> None:
    radius = float(np.sqrt(T))
    theta = np.linspace(0, 2 * np.pi, 400)
    bx = radius * np.cos(theta)
    by = radius * np.sin(theta)
    ax.fill(bx, by, color=color, alpha=0.08, zorder=1)
    ax.plot(bx, by, color=color, lw=2.0, ls="--", zorder=4)


def _set_axes(ax,
              axes_lim: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]) -> None:
    if axes_lim is not None:
        (xmin, xmax), (ymin, ymax) = axes_lim
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.grid(True, alpha=0.4)


def _save(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_latent_space(z: np.ndarray,
                      labels: np.ndarray,
                      label_names: Dict[int, str],
                      save_path: str | Path,
                      *,
                      mode: str = "training",
                      T: Optional[float] = None,
                      axes_lim: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                      figsize: Tuple[float, float] = (8, 8),
                      dpi: int = 200,
                      alpha: float = 0.6,
                      bf_color: str = "#3a6fb5",
                      unseen_color: str = "#c0392b") -> None:
    """Plot 2-D latent projections.

    Parameters
    ----------
    z           : (N, 2) array of g_psi(z) projections.
    labels      : (N,) integer labels.
    label_names : {label: display_name}.
    save_path   : output file. In 'unseen' mode the per-class suffix is
                  appended to the stem (e.g. 'foo.png' -> 'foo_<class>.png').
    mode        : 'training' or 'unseen'.
    T           : if provided, draws the dashed bona fide circle at sqrt(T).
    axes_lim    : optional ((xmin, xmax), (ymin, ymax)) frame.
    """
    z = np.asarray(z)
    labels = np.asarray(labels)
    save_path = Path(save_path)

    if mode == "training":
        _plot_training(z, labels, label_names, save_path, T,
                       axes_lim, figsize, dpi, alpha)
    elif mode == "unseen":
        _plot_unseen(z, labels, label_names, save_path, T,
                     axes_lim, figsize, dpi, alpha, bf_color, unseen_color)
    else:
        raise ValueError(f"mode must be 'training' or 'unseen', got '{mode}'.")


def _plot_training(z: np.ndarray,
                   labels: np.ndarray,
                   label_names: Dict[int, str],
                   save_path: Path,
                   T: Optional[float],
                   axes_lim,
                   figsize, dpi, alpha) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    for c in np.unique(labels):
        idx = labels == c
        name = label_names.get(int(c), f"class {c}")
        ax.scatter(z[idx, 0], z[idx, 1], label=name, alpha=alpha)
    if T is not None:
        _draw_threshold_circle(ax, T)
    _set_axes(ax, axes_lim)
    ax.legend(loc="best", framealpha=0.9)
    _save(fig, save_path, dpi)


def _plot_unseen(z: np.ndarray,
                 labels: np.ndarray,
                 label_names: Dict[int, str],
                 save_path: Path,
                 T: Optional[float],
                 axes_lim,
                 figsize, dpi, alpha,
                 bf_color, unseen_color) -> None:
    """Render one figure per unseen class. Bona fide samples (label 0)
    are overlaid on every figure if present."""
    bf_mask = labels == 0
    unseen_classes = [int(c) for c in np.unique(labels) if c != 0]

    for c in unseen_classes:
        cls_name = label_names.get(c, f"class_{c}")
        # Build per-class output path: foo.png -> foo_<cls>.png
        if "{name}" in str(save_path):
            out = Path(str(save_path).format(name=cls_name))
        else:
            out = save_path.with_name(f"{save_path.stem}_{cls_name}{save_path.suffix}")

        fig, ax = plt.subplots(figsize=figsize)

        if bf_mask.any():
            ax.scatter(z[bf_mask, 0], z[bf_mask, 1],
                       color=bf_color, alpha=alpha,
                       label=label_names.get(0, "bona fide"))

        cls_mask = labels == c
        ax.scatter(z[cls_mask, 0], z[cls_mask, 1],
                   color=unseen_color, alpha=alpha, label=cls_name)

        if T is not None:
            _draw_threshold_circle(ax, T)
        _set_axes(ax, axes_lim)
        ax.legend(loc="best", framealpha=0.9)
        _save(fig, out, dpi)
