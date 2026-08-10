"""Which representation's attribution actually predicts the APCER shifts?

`baselines_attribution.py` compares the diagnostic baselines on cross-seed
stability and on the entropy of the attribution they produce. The criterion
that decides which representation is useful, however, is whether that
attribution is *predictive of the ablation outcome*, and answering it needs the
exhaustive leave-one-material-out matrix.

Nothing is retrained here: the LOMO shifts are on disk and so are the
attributions. For each representation we pair p_{u,k} with Delta/sigma and score
it on the paper's own two axes (Sec. V-C):

  * directional : sign concordance among the anchors it calls primary (p >= 0.30)
  * selectivity : do the anchors it calls marginal (p < 0.30) stay inside the
                  noise floor of Section IV-D?

Everything is computed at CLUSTER level — one unit per (sensor, removed
material, unseen PAI), averaging the four audited systems, which share dataset,
splits and encoders and are therefore replicates rather than independent trials.

Usage
-----
    python scripts/baselines_predictiveness.py --save-json outputs/point_a_predictiveness.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.correlate_attr_apcer import (  # noqa: E402
    _spearman, attribution_matrix, shift_records)

from scripts._protocol import NOISE_FLOOR, P_THRESHOLD  # noqa: E402

# Representations read from the stored baselines JSONs (single seed) and those
# recomputed here from multi-seed g_psi checkpoints.
STORED = ["raw", "pca", "tsne", "umap"]
GEO_VARIANTS = {
    "xfpad":            "geometric_{scanner}_[0-9].pth",
    "xfpad_radialonly": "geometric_{scanner}_radialonly_[0-9].pth",
    "xfpad_cosface":    "geometric_{scanner}_cosface_[0-9].pth",
    "xfpad_arcface":    "geometric_{scanner}_arcface_[0-9].pth",
}


def _binom(k: int, n: int) -> float:
    if n == 0:
        return float("nan")
    kk = max(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(kk, n + 1)) / 2 ** n)


def _entropy(weights: List[float]) -> float:
    w = np.clip(np.asarray(weights, float), 1e-12, None)
    K = len(w)
    return float(-(w * np.log(w)).sum() / np.log(K)) if K > 1 else 0.0


def load_stored(scanners: List[str]) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """{rep: {scanner: {unseen: {material: p}}}} from outputs/baselines_*.json."""
    out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = defaultdict(dict)
    for sc in scanners:
        path = REPO / "outputs" / f"baselines_{sc}.json"
        if not path.exists():
            print(f"[{sc}] {path.name} missing — stored baselines skipped")
            continue
        d = json.load(path.open())
        for rep in STORED:
            if rep not in d:
                continue
            out[rep][sc] = {pai: {m: float(w) for m, w in v["ranked"]}
                            for pai, v in d[rep].items()}
    return out


def clusters(records: List[dict], attr: Dict[str, Dict[str, Dict[str, float]]]) -> List[dict]:
    """One unit per (sensor, removed material, unseen), backbones averaged."""
    acc: Dict[tuple, List[float]] = defaultdict(list)
    for r in records:
        acc[(r["scanner"], r["removed_material"], r["unseen"])].append(r["delta_over_sigma"])
    out = []
    for (sc, mat, u), vals in acc.items():
        p = attr.get(sc, {}).get(u, {}).get(mat)
        if p is None:
            continue
        out.append({"scanner": sc, "removed_material": mat, "unseen": u,
                    "p_uk": float(p), "effect": float(np.mean(vals))})
    return out


def score(cl: List[dict]) -> dict:
    if not cl:
        return {}
    p = [c["p_uk"] for c in cl]; e = [c["effect"] for c in cl]
    direc = [c for c in cl if c["p_uk"] >= P_THRESHOLD]
    marg = [c for c in cl if c["p_uk"] < P_THRESHOLD]
    hits = sum(c["effect"] > 0 for c in direc)
    return {
        "n_clusters": len(cl),
        "spearman": _spearman(p, e),
        "directional_n": len(direc),
        "directional_hits": hits,
        "directional_p": _binom(hits, len(direc)),
        "directional_median_effect": float(np.median([abs(c["effect"]) for c in direc]))
        if direc else float("nan"),
        "marginal_n": len(marg),
        "marginal_inside_floor": float(np.mean([abs(c["effect"]) < NOISE_FLOOR for c in marg]))
        if marg else float("nan"),
        "marginal_median_effect": float(np.median([abs(c["effect"]) for c in marg]))
        if marg else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Predictiveness of the diagnostic baselines.")
    ap.add_argument("--scanners", nargs="+", default=["greenbit", "dermalog"])
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    records: List[dict] = []
    for sc in args.scanners:
        records += shift_records(sc)
    if not records:
        print("No LOMO JSONs found in outputs/lomo/.")
        return

    reps: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = dict(load_stored(args.scanners))
    entropies: Dict[str, float] = {}

    for name, glob in GEO_VARIANTS.items():
        per_scanner = {}
        for sc in args.scanners:
            try:
                per_scanner[sc] = attribution_matrix(sc, glob, args.tau)
            except FileNotFoundError:
                print(f"[{name}/{sc}] no checkpoints — skipped")
        if per_scanner:
            reps[name] = per_scanner

    for name, per_scanner in reps.items():
        vals = [_entropy(list(w.values()))
                for sc in per_scanner for w in per_scanner[sc].values()]
        entropies[name] = float(np.mean(vals)) if vals else float("nan")

    print("=" * 92)
    print("DIAGNOSTIC BASELINES — does the attribution PREDICT the ablation outcome?  (cluster level)")
    print("=" * 92)
    print(f"{'representation':<18} {'H':>5} {'rho':>7} | {'directional p>=0.30':>21} "
          f"| {'selectivity (p<0.30)':>22}")
    print(f"{'':<18} {'':>5} {'':>7} | {'hits':>8} {'binom p':>11} "
          f"| {'inside floor':>13} {'median':>8}")
    print("-" * 92)
    results = {}
    for name in ["raw", "pca", "tsne", "umap", "xfpad_radialonly", "xfpad_cosface",
                 "xfpad_arcface", "xfpad"]:
        if name not in reps:
            continue
        st = score(clusters(records, reps[name]))
        if not st:
            continue
        st["entropy"] = entropies.get(name)
        results[name] = st
        print(f"{name:<18} {entropies.get(name, float('nan')):>5.2f} "
              f"{st['spearman']:>+7.3f} | "
              f"{st['directional_hits']:>3}/{st['directional_n']:<4} "
              f"{st['directional_p']:>11.4f} | "
              f"{100*st['marginal_inside_floor']:>12.1f}% {st['marginal_median_effect']:>8.2f}")
    print("-" * 92)
    print(f"H = mean normalised entropy (lower = more decisive); rho = Spearman(p, Delta/sigma);")
    print(f"selectivity = share of marginal anchors inside the |Delta/sigma| < {NOISE_FLOOR} noise floor.")
    print("NB: raw/pca/tsne/umap come from the stored single-seed baselines;")
    print("    the xfpad* rows average their available g_psi seeds.")

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, out.open("w"), indent=2)
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
