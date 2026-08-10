"""Quantify the correlation between attribution and APCER shift.

It tests whether the X-FPAD attribution weights p_{u,k} predict the APCER
shift observed when the anchor material k is removed (leave-one-material-out,
Phase 3a).

For every (sensor, backbone, removed material k, unseen PAI u) it pairs:
  * p_{u,k}  — X-FPAD attribution of unseen u to the removed material k,
               recomputed here from the canonical g_psi checkpoints (mean over
               the available seeds) via the same estimator as the paper;
  * Delta/sigma — APCER shift of u when k is ablated, from the Phase 3 JSONs
               (outputs/lomo/), Delta against the baseline, sigma the pooled
               std across baseline+ablated runs.

Reported statistics, in two tiers:
  PRIMARY  — directional: sign-concordance of Delta/sigma with the p>=0.30
             prediction (binomial test), and Spearman rank correlation.
  SECONDARY— global Pearson r of (p_{u,k}, Delta/sigma), with the Dermalog
             magnitude inversion discussed as an absorption effect, not a
             failure of the ranking.

Prereqs: the LOMO block of scripts/lomo_driver.py has produced the baseline
and per-material JSONs, and the canonical g_psi checkpoints exist
(checkpoints/geometric_<scanner>_<seed>.pth or _1_0_<seed> — see --geo-glob).

Usage
-----
    python scripts/correlate_attr_apcer.py --save-json outputs/point_b_correlation.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts._protocol import P_THRESHOLD  # noqa: E402
from scripts._common import (  # noqa: E402
    bona_fide_label,
    known_names,
    known_pairs,
    unseen_names,
    unseen_pairs,
)
from scripts.lomo_driver import AUDITED, LOMO_CELLS  # noqa: E402
from xfpad.config import load_config  # noqa: E402
from xfpad.data import build_labels  # noqa: E402
from xfpad.metrics import analyze_unseen_pais  # noqa: E402
from xfpad.utils import features_path, read_split, split_path  # noqa: E402

LOMO_OUT = REPO / "outputs" / "lomo"


# ---------------------------------------------------------------------------
# Attribution p_{u,k}: recompute from canonical g_psi checkpoints
# ---------------------------------------------------------------------------

def _project(geo_ckpt: Path, feats: np.ndarray, dropout: float) -> np.ndarray:
    import torch
    from xfpad.models import GeometricEncoder
    model = GeometricEncoder(dropout=dropout)
    model.load_state_dict(torch.load(geo_ckpt, map_location="cpu")["model"])
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(feats).float()).cpu().numpy()


def attribution_matrix(scanner: str, geo_glob: str,
                       tau: float) -> Dict[str, Dict[str, float]]:
    """Return {unseen_name: {material_name: mean p_{u,k} over seeds}}."""
    cfg = load_config(str(REPO / "configs" / f"{scanner}.yaml"))
    feats_tr = np.load(features_path(cfg.paths.features_dir, scanner, "train"))
    feats_te = np.load(features_path(cfg.paths.features_dir, scanner, "test"))
    y_tr = np.array(build_labels(
        read_split(split_path(cfg.paths.splits_dir, scanner, "train")),
        known_pairs(cfg)))
    y_te = np.array(build_labels(
        read_split(split_path(cfg.paths.splits_dir, scanner, "test")),
        unseen_pairs(cfg)))
    tr_names, te_names = known_names(cfg), unseen_names(cfg)
    bf = bona_fide_label(tr_names)
    proto_order = [tr_names[l] for l in sorted(tr_names) if l != bf]

    ckpts = sorted(glob.glob(str(REPO / "checkpoints" / geo_glob.format(scanner=scanner))))
    if not ckpts:
        raise FileNotFoundError(
            f"No canonical g_psi checkpoints match "
            f"'{geo_glob.format(scanner=scanner)}' in checkpoints/.")

    acc: Dict[str, List[np.ndarray]] = {}
    for ck in ckpts:
        z_tr = _project(Path(ck), feats_tr, cfg.geometric.dropout)
        z_te = _project(Path(ck), feats_te, cfg.geometric.dropout)
        res = analyze_unseen_pais(
            z_tr, y_tr, tr_names, z_te, y_te, te_names,
            prototype_order=proto_order, bf_key=tr_names[bf], tau=tau)
        for u, r in res.items():
            w = dict(r["ranked_anchors"])
            acc.setdefault(u, []).append(np.array([w[m] for m in proto_order]))

    out: Dict[str, Dict[str, float]] = {}
    for u, mats in acc.items():
        mean_w = np.mean(mats, axis=0)
        out[u] = {m: float(mean_w[i]) for i, m in enumerate(proto_order)}
    return out


# ---------------------------------------------------------------------------
# APCER shifts from Phase 3 JSONs
# ---------------------------------------------------------------------------

def _load_per_pai(path: Path) -> Dict[str, Tuple[float, float, int]]:
    """Return {unseen_name: (mean, std, n)} from a phase3 JSON."""
    with path.open() as f:
        data = json.load(f)
    per = data["aggregated"]["per_pai"]
    return {k: (v["mean"], v["std"], v["n"]) for k, v in per.items()}


def shift_records(scanner: str) -> List[dict]:
    """For each (backbone, removed material, unseen PAI): Delta and Delta/sigma."""
    records: List[dict] = []
    materials = [(name, mat) for s, _sub, name, mat in LOMO_CELLS if s == scanner]
    for backbone in AUDITED:
        base_p = LOMO_OUT / f"phase3_{scanner}_{backbone}_baseline.json"
        if not base_p.exists():
            continue
        base = _load_per_pai(base_p)
        for abl_name, material in materials:
            abl_p = LOMO_OUT / f"phase3_{scanner}_{backbone}_{abl_name}.json"
            if not abl_p.exists():
                continue
            abl = _load_per_pai(abl_p)
            for u in base:
                if u not in abl:
                    continue
                m0, s0, n0 = base[u]
                m1, s1, n1 = abl[u]
                pooled = math.sqrt(((n0 - 1) * s0 ** 2 + (n1 - 1) * s1 ** 2)
                                   / max(n0 + n1 - 2, 1)) if (n0 + n1) > 2 else s0
                delta = m1 - m0
                records.append({
                    "scanner": scanner, "backbone": backbone,
                    "removed_material": material, "unseen": u,
                    "delta": delta,
                    "delta_over_sigma": delta / pooled if pooled > 1e-9 else 0.0,
                })
    return records


# ---------------------------------------------------------------------------
# Correlation statistics
# ---------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _binom_sign_test(k: int, n: int) -> float:
    """Two-sided binomial tail prob under p=0.5 (sign concordance)."""
    if n == 0:
        return float("nan")
    from math import comb
    kk = max(k, n - k)
    tail = sum(comb(n, i) for i in range(kk, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def correlate(records: List[dict], attr: Dict[str, Dict[str, Dict[str, float]]],
              p_threshold: float) -> dict:
    p_vals, eff_vals = [], []
    directional_hits = directional_tot = 0
    for r in records:
        a = attr.get(r["scanner"], {}).get(r["unseen"], {})
        p = a.get(r["removed_material"])
        if p is None:
            continue
        r["p_uk"] = p
        p_vals.append(p); eff_vals.append(r["delta_over_sigma"])
        if p >= p_threshold:
            directional_tot += 1
            if r["delta_over_sigma"] > 0:
                directional_hits += 1
    p_arr, eff_arr = np.array(p_vals), np.array(eff_vals)
    return {
        "n_pairs": len(p_vals),
        "pearson_r": _pearson(p_arr, eff_arr),
        "spearman_rho": _spearman(p_arr, eff_arr),
        "directional": {
            "threshold": p_threshold,
            "n": directional_tot, "positive": directional_hits,
            "concordance": directional_hits / directional_tot
            if directional_tot else float("nan"),
            "binom_p": _binom_sign_test(directional_hits, directional_tot),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Correlation between attribution and APCER shift.")
    ap.add_argument("--geo-glob", default="geometric_{scanner}_[0-9].pth",
                    help="Glob (with {scanner}) for canonical g_psi checkpoints "
                         "used to recompute p_{u,k}. Default matches the "
                         "multi-seed canonical encoders.")
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--p-threshold", type=float, default=P_THRESHOLD,
                    help="p_{u,k} threshold for a directional prediction.")
    ap.add_argument("--scanners", nargs="+", default=["greenbit", "dermalog"])
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    attr: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_records: List[dict] = []
    for sc in args.scanners:
        attr[sc] = attribution_matrix(sc, args.geo_glob, args.tau)
        all_records += shift_records(sc)

    if not all_records:
        print("No Phase 3 LOMO JSONs found in outputs/lomo/. Run lomo_driver "
              "first (baseline + lomo blocks).")
        return

    overall = correlate(all_records, attr, args.p_threshold)
    per_backbone = {
        bb: correlate([r for r in all_records if r["backbone"] == bb],
                      attr, args.p_threshold)
        for bb in AUDITED
    }

    print("=" * 66)
    print(f"ATTRIBUTION vs APCER SHIFT   (n={overall['n_pairs']} pairs)")
    print("=" * 66)
    print(f"  PRIMARY  Spearman rho = {overall['spearman_rho']:+.3f}")
    d = overall["directional"]
    print(f"  PRIMARY  directional (p>={d['threshold']}): "
          f"{d['positive']}/{d['n']} correct sign "
          f"(concordance {d['concordance']:.2f}, binom p={d['binom_p']:.1e})")
    print(f"  SECONDARY global Pearson r = {overall['pearson_r']:+.3f}")
    print("  per-backbone Spearman:")
    for bb, st in per_backbone.items():
        print(f"    {bb:<14} rho={st['spearman_rho']:+.3f}  "
              f"(n={st['n_pairs']}, dir {st['directional']['positive']}/"
              f"{st['directional']['n']})")

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump({"overall": overall, "per_backbone": per_backbone,
                       "records": all_records}, f, indent=2)
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
