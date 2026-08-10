"""Redraw the Phase 2 panels with the training manifold behind them.

An unseen PAI drawn on empty axes shows where the cluster landed, not what it
landed on: the angular sector it occupies -- the quantity the whole of Phase 2
is about -- is not readable without the anchors beside it.

Nothing is retrained: `phase2_map_unseen.py` stores the projections in
outputs/projections/<scanner>/seed<N>.npz, so they are reloaded and redrawn.

Usage
-----
    python scripts/replot_phase2.py
    python scripts/replot_phase2.py --selftest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from xfpad.viz import plot_latent_space  # noqa: E402

SCANNERS = ("greenbit", "dermalog")
OUT_DIR = REPO / "patch_bg"


def _read_yaml(path: Path) -> dict:
    """Not every config is UTF-8: some were saved as cp1252."""
    for enc in ("utf-8", "cp1252"):
        try:
            return yaml.safe_load(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"{path}: ne' UTF-8 ne' cp1252")


def names_from_config(scanner: str) -> tuple[dict[int, str], dict[int, str], float]:
    """(training class names, unseen class names, bona fide radius).

    The per-sensor configs are overlays: the labels live there, rho_bf lives in
    base.yaml. The figures redrawn here are the rho_bf = 1.0 ones (the '1_0'
    in the file names), so the value read is checked rather than assumed.
    """
    cfg = _read_yaml(REPO / "configs" / f"{scanner}_cons.yaml")
    blocks = [cfg[k] for k in ("training_labels", "unseen_labels")]
    train_names = {int(k): str(v) for k, v in blocks[0]["names"].items()}
    unseen_names = {int(k): str(v) for k, v in blocks[1]["names"].items()}

    base = _read_yaml(REPO / "configs" / "base.yaml")
    loss = next(v for v in base.values() if isinstance(v, dict) and "rho_bf" in v)
    rho_bf = float(loss["rho_bf"])
    assert rho_bf == 1.0, f"the expected figures are the rho_bf=1.0 ones, read {rho_bf}"
    return train_names, unseen_names, rho_bf


def replot(scanner: str, seed: int, out_dir: Path, background: bool = False) -> list[Path]:
    npz = REPO / "outputs" / "projections" / scanner / f"seed{seed}.npz"
    assert npz.exists(), f"projections missing: {npz}"
    d = np.load(npz)
    z_train, y_train = d["z_train"], d["labels_train"]
    z_test, y_test = d["z_test"], d["labels_test"]

    train_names, unseen_names, rho_bf = names_from_config(scanner)
    assert set(np.unique(y_train)) <= set(train_names), \
        f"{scanner}: training labels with no name in the config"
    assert set(np.unique(y_test)) <= set(unseen_names), \
        f"{scanner}: unseen labels with no name in the config"

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"latent_space_{scanner}_1_0"
    T = rho_bf ** 2

    # A square frame centred on the origin, shared by every panel. Square
    # because the geometry is radial-angular and a squashed box distorts the
    # angles to the eye; shared because panels drawn on different frames cannot
    # be compared, and comparison is the point of Phase 2.
    r = 1.05 * float(max(np.abs(z_train).max(), np.abs(z_test).max()))
    lims = ((-r, r), (-r, r))

    plot_latent_space(z_train, y_train, train_names,
                      out_dir / f"{stem}.png", mode="training", T=T,
                      axes_lim=lims, figsize=(8, 8), dpi=200, alpha=0.6)

    # The manifold background is available but off by default: on a shared
    # square frame the angular position of the unseen cluster already reads
    # off the axes, and redrawing the known materials under each panel adds
    # only clutter and a doubled legend.
    plot_latent_space(z_test, y_test, unseen_names,
                      out_dir / f"{stem}.png", mode="unseen", T=T,
                      axes_lim=lims, figsize=(8, 8), dpi=200, alpha=0.75,
                      bg=(z_train, y_train, train_names) if background else None)

    produced = sorted(out_dir.glob(f"{stem}*.png"))
    expected = 1 + len([c for c in np.unique(y_test) if c != 0])
    assert len(produced) == expected, \
        f"{scanner}: expected {expected} figures, produced {len(produced)}"
    return produced


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--background", action="store_true",
                    help="draw the training manifold faintly under each unseen PAI")
    ap.add_argument("--selftest", action="store_true",
                    help="check names, shared frame and counts without writing to patch_bg/")
    args = ap.parse_args()

    out = Path(args.out)
    if args.selftest:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for sc in SCANNERS:
                files = replot(sc, args.seed, Path(tmp) / sc, args.background)
                print(f"  [{sc}] {len(files)} figures produced")
        print("  selftest ok, nothing written")
        return

    for sc in SCANNERS:
        files = replot(sc, args.seed, out, args.background)
        print(f"  [{sc}] {len(files)} figures -> {out}")


if __name__ == "__main__":
    main()
