"""Phase 3a figure: attribution weight against measured effect.

A single panel carrying the two readings of the experiment:

  * filled markers are the cells Phase 2 named as primary anchors
    (p_{u,k} >= 0.30). They are the targeted ablation: a prediction made before
    the outcome was known and checked prospectively.
  * hollow markers are every other cell of the exhaustive sweep. They are the
    specificity control: were the effect to appear there too, it would not be
    attributable to the anchor but to the removal of data as such.

The horizontal band is the noise floor of Section IV-D, the level the paper
reads the ablation table with.

The units are the clusters (sensor, removed material, unseen PAI), with the four
audited systems averaged: they share data, splits, features and encoder, so they
are replicates rather than independent trials.

Usage
-----
    python scripts/plot_phase3_scatter.py
    python scripts/plot_phase3_scatter.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "outputs" / "point_b_correlation.json"
OUT = REPO / "patch_bg" / "phase3_scatter.png"

sys.path.insert(0, str(REPO))
from scripts._protocol import AUDITED, NOISE_FLOOR, P_THRESHOLD  # noqa: E402


def clusters(records: list[dict]) -> list[dict]:
    """One unit per (scanner, removed material, unseen PAI)."""
    acc: dict[tuple, list[float]] = defaultdict(list)
    pw: dict[tuple, float] = {}
    for r in records:
        key = (r["scanner"], r["removed_material"], r["unseen"])
        acc[key].append(float(r["delta_over_sigma"]))
        pw[key] = float(r["p_uk"])
    out = []
    for key, vals in acc.items():
        assert len(vals) == len(AUDITED), \
            f"{key}: expected {len(AUDITED)} audited systems, found {len(vals)}"
        out.append({"scanner": key[0], "removed": key[1], "unseen": key[2],
                    "p_uk": pw[key], "effect": float(np.mean(vals)),
                    "per_system": [float(v) for v in vals]})
    return out


def build(save: Path | None) -> list[dict]:
    recs = json.loads(SRC.read_text(encoding="utf-8"))["records"]
    cl = clusters(recs)
    assert len(cl) == 58, f"expected 58 cluster units, found {len(cl)}"

    direc = [c for c in cl if c["p_uk"] >= P_THRESHOLD]
    marg = [c for c in cl if c["p_uk"] < P_THRESHOLD]
    assert all(c["effect"] > 0 for c in direc), \
        "a directional prediction has a negative sign: review before publishing"

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.axhspan(-NOISE_FLOOR, NOISE_FLOOR, color="0.85", zorder=0)
    ax.axhline(0.0, color="0.45", lw=0.6, zorder=1)
    ax.axvline(P_THRESHOLD, color="0.45", lw=0.6, ls=":", zorder=1)

    # One unit reaches +16.0 and on its own would compress everything else into
    # the lower quarter of the panel. The axis is capped and that unit marked
    # with a triangle on the edge, rather than losing the other 57 to it.
    y_cap = 7.5
    for group, style in ((marg, dict(facecolors="none", edgecolors="#3a6fb5",
                                     label="marginal anchors (control)")),
                         (direc, dict(facecolors="#c0392b", edgecolors="#c0392b",
                                      label="predicted anchors"))):
        x = np.array([c["p_uk"] for c in group])
        y = np.array([c["effect"] for c in group])
        over = y > y_cap
        ax.scatter(x[~over], y[~over], s=26, linewidths=0.9, zorder=3, **style)
        if over.any():
            ax.scatter(x[over], np.full(over.sum(), y_cap), s=34, marker="^",
                       linewidths=0.9, zorder=3,
                       **{k: v for k, v in style.items() if k != "label"})
            for xi, yi in zip(x[over], y[over]):
                ax.annotate(f"{yi:+.1f}", (xi, y_cap), textcoords="offset points",
                            xytext=(0, -9), ha="center", fontsize=6.5, color="#c0392b")

    ax.set_xlabel(r"attribution weight $p_{u,k}$")
    ax.set_ylabel(r"effect size $\Delta/\sigma$")
    ax.set_xlim(-0.02, 0.92)
    ax.set_ylim(-2.6, y_cap + 0.6)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9, borderpad=0.3,
              handletextpad=0.4)
    fig.tight_layout(pad=0.2)

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=400, bbox_inches="tight")
    plt.close(fig)

    inside = lambda g: 100 * np.mean([abs(c["effect"]) < NOISE_FLOOR for c in g])
    print(f"  cluster units               {len(cl)}")
    print(f"  predicted (p >= {P_THRESHOLD})      {len(direc)}, all correctly signed, "
          f"{inside(direc):.1f}% inside the noise floor")
    print(f"  control (p < {P_THRESHOLD})         {len(marg)}, "
          f"{inside(marg):.1f}% inside the noise floor")
    return cl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--selftest", action="store_true",
                    help="check counts and signs without writing the figure")
    args = ap.parse_args()
    if args.selftest:
        cl = build(None)
        # The spread across systems is not drawn -- a min-max over four fixed
        # architectures is not a distribution, and the sigma of the runs is
        # already the denominator of Delta/sigma -- but it is checked: were the
        # systems to split on a predicted unit, the figure would be averaging a
        # disagreement.
        direc = [c for c in cl if c["p_uk"] >= P_THRESHOLD]
        ev = [(c, v) for c in direc for v in c["per_system"]]
        ok = sum(1 for _c, v in ev if v > 0)
        assert ok >= len(ev) - 1, f"{len(ev)-ok} evaluations of the opposite sign"
        assert all(sum(1 for v in c["per_system"] if v > 0) >= 3 for c in direc), \
            "a predicted unit has half of the systems disagreeing"
        print(f"  per-system evaluations      {ok}/{len(ev)} correctly signed")
        print("  selftest ok, no file written")
    else:
        build(Path(args.out))
        print(f"  figure -> {args.out}")


if __name__ == "__main__":
    main()
