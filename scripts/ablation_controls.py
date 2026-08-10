"""Is the APCER shift anchor-specific, or just "less training data"?

Removing a material also changes training-set size, class balance and
optimisation dynamics, so an observed shift need not come from the geometric
anchor relation. This compares the ablations the manifold implicates with the
controls it does not.

Three groups are compared, all at CLUSTER level (one unit per sensor x unseen
PAI, averaging the audited systems):

  directional    the removed material is a primary/co-dominant anchor of that
                 unseen PAI (p_{u,k} >= 0.30) -- where the paper predicts an effect
  marginal       the removed material is a marginal anchor (p_{u,k} < 0.30)
                 -- same intervention, no geometric relation
  random_n<N>    N spoof samples removed at random, N matched to the size of an
                 anchor material -- same quantity of data, no material identity

If the effect were driven by data quantity, the random controls would look like
the directional ablations. If it is geometric, only the directional group moves.

Usage
-----
    python scripts/ablation_controls.py --save-json outputs/point_c_controls.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.correlate_attr_apcer import attribution_matrix  # noqa: E402
from scripts._protocol import AUDITED, NOISE_FLOOR, P_THRESHOLD  # noqa: E402

LOMO = REPO / "outputs" / "lomo"


def _per_pai(path: Path) -> Dict[str, tuple]:
    d = json.load(path.open())["aggregated"]["per_pai"]
    return {k: (v["mean"], v["std"], v["n"]) for k, v in d.items()}


def _shift(base: tuple, abl: tuple) -> float:
    m0, s0, n0 = base
    m1, s1, n1 = abl
    pooled = math.sqrt(((n0 - 1) * s0 ** 2 + (n1 - 1) * s1 ** 2)
                       / max(n0 + n1 - 2, 1)) if (n0 + n1) > 2 else s0
    return (m1 - m0) / pooled if pooled > 1e-9 else 0.0


def collect(scanner: str, tau: float) -> List[dict]:
    """One record per (cell, unseen PAI), Delta/sigma averaged over backbones."""
    attr = attribution_matrix(scanner, "geometric_{scanner}_[0-9].pth", tau)
    cells: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for bb in AUDITED:
        base_p = LOMO / f"phase3_{scanner}_{bb}_baseline.json"
        if not base_p.exists():
            continue
        base = _per_pai(base_p)
        for p in sorted(LOMO.glob(f"phase3_{scanner}_{bb}_*.json")):
            cell = p.stem.replace(f"phase3_{scanner}_{bb}_", "")
            if cell == "baseline" or cell.startswith("phase4"):
                continue
            abl = _per_pai(p)
            for u in base:
                if u in abl:
                    cells[cell][u].append(_shift(base[u], abl[u]))

    out: List[dict] = []
    for cell, per_pai in cells.items():
        for u, vals in per_pai.items():
            rec = {"scanner": scanner, "cell": cell, "unseen": u,
                   "effect": float(np.mean(vals)), "n_backbones": len(vals)}
            if cell.startswith("random_n"):
                rec["group"] = cell
            elif cell.startswith("without_"):
                mat = None
                for m in attr.get(u, {}):
                    slug = m.lower().replace(" ", "_").replace("'", "")
                    if cell == f"without_{slug}":
                        mat = m
                        break
                if mat is None:
                    continue
                rec["removed_material"] = mat
                rec["p_uk"] = float(attr[u][mat])
                rec["group"] = "directional" if rec["p_uk"] >= P_THRESHOLD else "marginal"
            else:
                continue
            out.append(rec)
    return out


def _mannwhitney_perm(a: List[float], b: List[float], n_perm: int = 20000,
                      seed: int = 0) -> float:
    """One-sided permutation p for median(b) > median(a)."""
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    obs = np.median(b) - np.median(a)
    pool = list(a) + list(b); rng = random.Random(seed); cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if (np.median(pool[len(a):]) - np.median(pool[:len(a)])) >= obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def selftest(recs: List[dict]) -> None:
    """The three numbers Section V-C and Section IV-D of the paper quote."""
    by = defaultdict(list)
    for r in recs:
        by[r["group"]].append(abs(r["effect"]))

    ctrl = [v for g, vs in by.items() if g.startswith("random") for v in vs]
    assert len(ctrl) == 16, len(ctrl)
    top = max(ctrl)
    # The floor is calibrated on these: it has to sit above all of them, and
    # the paper quotes their largest excursion.
    assert round(top, 2) == 1.16, top
    assert top < NOISE_FLOOR, (top, NOISE_FLOOR)

    d, m = by["directional"], by["marginal"]
    assert len(d) == 11 and len(m) == 47, (len(d), len(m))
    assert sum(x >= NOISE_FLOOR for x in d) == 8, d
    assert round(100 * np.mean([x < NOISE_FLOOR for x in m]), 1) == 89.4, m

    print(f"  selftest ok — {len(ctrl)} matched-size control units peak at "
          f"{top:.2f} sigma, below the {NOISE_FLOOR} floor; "
          f"{sum(x >= NOISE_FLOOR for x in d)}/11 predicted units above it, "
          f"{100 * np.mean([x < NOISE_FLOOR for x in m]):.1f}% of the 47 inside")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablation confound controls.")
    ap.add_argument("--scanners", nargs="+", default=["greenbit", "dermalog"])
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--save-json", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    recs: List[dict] = []
    for sc in args.scanners:
        recs += collect(sc, args.tau)
    if not recs:
        print("No LOMO JSONs found.")
        return

    if args.selftest:
        selftest(recs)
        return

    groups: Dict[str, List[float]] = defaultdict(list)
    for r in recs:
        groups[r["group"]].append(abs(r["effect"]))

    print("=" * 84)
    print("CONFOUND CONTROLS — is the shift anchor-specific or quantity-driven?  (cluster level)")
    print("=" * 84)
    print(f"{'group':<22} {'n':>4} {'median |d/s|':>13} {'mean':>8} "
          f"{'inside floor':>14} {'max':>7}")
    print("-" * 84)
    order = ["directional", "marginal"] + sorted(g for g in groups if g.startswith("random_n"))
    for g in order:
        v = groups.get(g)
        if not v:
            continue
        print(f"{g:<22} {len(v):>4} {np.median(v):>13.2f} {np.mean(v):>8.2f} "
              f"{100*np.mean([x < NOISE_FLOOR for x in v]):>13.1f}% {max(v):>7.2f}")
    print("-" * 84)

    dire = groups.get("directional", [])
    print("permutation tests (one-sided, median):")
    for g in order[1:]:
        if g in groups and dire:
            p = _mannwhitney_perm(groups[g], dire)
            print(f"   directional > {g:<18} p = {p:.4f}")

    print("\nnegative controls — marginal anchors, per sensor "
          f"(share inside the |d/s| < {NOISE_FLOOR} floor):")
    per_mat: Dict[tuple, List[float]] = defaultdict(list)
    for r in recs:
        if r.get("group") == "marginal":
            per_mat[(r["scanner"], r["removed_material"])].append(abs(r["effect"]))
    for (sc, mat), v in sorted(per_mat.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        print(f"   {sc:<9} without {mat:<13} {sum(x < NOISE_FLOOR for x in v)}/{len(v)} "
              f"  median {np.median(v):.2f}")

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"records": recs,
                   "groups": {g: {"n": len(v), "median": float(np.median(v)),
                                  "mean": float(np.mean(v)),
                                  "inside_floor": float(np.mean([x < NOISE_FLOOR for x in v]))}
                              for g, v in groups.items()}}, out.open("w"), indent=2)
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
