"""Phase 3b analysis — manifold-guided dataset composition.

Given the Phase 3b cells produced by `lomo_driver.py --blocks phase4`
(one reduced-vocabulary baseline plus one cell per re-added candidate), this
script applies the five selection strategies and scores them by how much of the
unseen-APCER gap each one closes:

    Recovery(u, candidate) = APCER_reduced(u) - APCER_reduced+candidate(u)

Strategies (all choose ONE candidate per unseen PAI, from the same pool, and
therefore reuse the same trained cells — no extra GPU time):

  xfpad         argmax_c p_{u,c} on the manifold of the REDUCED vocabulary
  raw_nc        argmax_c cos(u, centroid_c) in the raw 1280-D backbone space
  most_diff     argmin_c cos(u, centroid_c)  — the naive "cover the worst case"
  matched_size  the candidate whose sample count is closest to the xfpad pick,
                excluding the xfpad pick itself (isolates size from geometry)
  random        expectation over candidates (mean recovery), i.e. what an
                uninformed choice yields on average

PRIMARY result (robust to single-pick luck): the rank correlation between the
predicted attribution p_{u,c} and the OBSERVED recovery(u,c), over every
(unseen, candidate) pair — the forward analog of the ablation correlation.
Because the 'random' arm already needs every candidate's recovery, this whole
ranking is computed for free. A single hand-picked reduced vocabulary (chosen by an outcome-independent
rule: keep the fully-sampled base materials, candidates = the under-sampled ones)
is enough, since generalisation across materials is already established
exhaustively by the leave-one-material-out sweep.

COROLLARY: xfpad's top pick vs the direction-agnostic arms (random,
matched_size, most_diff). raw_nc is reported for transparency: the
baseline comparison showed it agrees with X-FPAD on the top-1 anchor, so a near-tie there is the expected and
honest outcome — it says the 2-D interpretable manifold loses no selection
information relative to the 1280-D space.

Usage
-----
    python scripts/phase4_analysis.py --save-json outputs/phase4_analysis.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts._common import (  # noqa: E402
    bona_fide_label, known_names, known_pairs, unseen_names, unseen_pairs,
)
# The JSON field stays named 'backbone' because that is what
# phase4_shared_anchor.py and the emitters read.
from scripts._protocol import AUDITED as BACKBONES  # noqa: E402
from scripts.lomo_driver import PHASE4_SETUP  # noqa: E402
from scripts.correlate_attr_apcer import _spearman  # noqa: E402
from xfpad.config import load_config  # noqa: E402
from xfpad.data import build_labels  # noqa: E402
from xfpad.metrics import analyze_unseen_pais  # noqa: E402
from xfpad.utils import features_path, read_split, split_path  # noqa: E402

LOMO_OUT = REPO / "outputs" / "lomo"
STRATEGIES = ["xfpad", "raw_nc", "random", "matched_size", "most_diff"]


def _slug(material: str) -> str:
    return material.lower().replace(" ", "_").replace("'", "")


def _per_pai(path: Path) -> Dict[str, float]:
    d = json.load(path.open())["aggregated"]["per_pai"]
    return {k: v["mean"] for k, v in d.items()}


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def _centroid_scores(scanner: str, candidates: List[str]) -> Dict[str, Dict[str, float]]:
    """cos(unseen cluster, candidate centroid) in the RAW 1280-D space.

    Returns {unseen: {candidate: cosine}}. Directions are taken from the bona
    fide centroid, matching the attribution estimator's convention.
    """
    cfg = load_config(str(REPO / "configs" / f"{scanner}.yaml"))
    ftr = np.load(features_path(cfg.paths.features_dir, scanner, "train"))
    fte = np.load(features_path(cfg.paths.features_dir, scanner, "test"))
    ytr = np.array(build_labels(
        read_split(split_path(cfg.paths.splits_dir, scanner, "train")), known_pairs(cfg)))
    yte = np.array(build_labels(
        read_split(split_path(cfg.paths.splits_dir, scanner, "test")), unseen_pairs(cfg)))
    tr_names, te_names = known_names(cfg), unseen_names(cfg)
    bf = bona_fide_label(tr_names)
    bf_c = np.median(ftr[ytr == bf], axis=0)

    name2lbl = {v: k for k, v in tr_names.items()}
    cand_dir = {}
    for c in candidates:
        d = np.median(ftr[ytr == name2lbl[c]], axis=0) - bf_c
        cand_dir[c] = d / max(np.linalg.norm(d), 1e-12)

    out: Dict[str, Dict[str, float]] = {}
    for lbl, uname in te_names.items():
        if uname == tr_names[bf]:
            continue
        m = yte == int(lbl)
        if not m.any():
            continue
        u = np.median(fte[m], axis=0) - bf_c
        u = u / max(np.linalg.norm(u), 1e-12)
        out[uname] = {c: float(u @ v) for c, v in cand_dir.items()}
    return out


def _xfpad_scores(scanner: str, candidates: List[str],
                  geo_glob: str) -> Dict[str, Dict[str, float]] | None:
    """p_{u,c} for the CANDIDATE materials, on the reduced-vocabulary manifold.

    The candidates are by construction *absent* from the reduced vocabulary —
    they are precisely the materials g_psi was not trained on — so they are not
    among its K prototypes and cannot be read off `ranked_anchors`. Their
    directions are therefore built the same way the paper builds a prototype
    (Eq. 5: unit vector from the bona fide centroid to the material's training
    centroid), but from their training samples pushed through the reduced
    encoder, which never saw them. This is the out-of-vocabulary analogue of
    p_{u,k} and the exact 2-D counterpart of `_centroid_scores`, making the
    xfpad/raw_nc comparison a like-for-like test of what the 2-D projection
    keeps. Attribution is defined identically in both spaces (see
    xfpad/metrics/attribution.py).

    Requires encoders trained on the reduced vocabulary (see
    scripts/train_reduced_encoder.py). Returns None if none are present.
    """
    import torch
    from xfpad.models import GeometricEncoder

    cks = sorted(glob.glob(str(REPO / "checkpoints" / geo_glob.format(scanner=scanner))))
    if not cks:
        return None
    cfg = load_config(str(REPO / "configs" / f"{scanner}.yaml"))
    full_names = known_names(cfg)
    bf_full = bona_fide_label(full_names)
    name2lbl = {v: k for k, v in full_names.items()}

    ftr_all = np.load(features_path(cfg.paths.features_dir, scanner, "train"))
    fte = np.load(features_path(cfg.paths.features_dir, scanner, "test"))
    ytr_all = np.array(build_labels(
        read_split(split_path(cfg.paths.splits_dir, scanner, "train")), known_pairs(cfg)))
    yte = np.array(build_labels(
        read_split(split_path(cfg.paths.splits_dir, scanner, "test")), unseen_pairs(cfg)))

    # Centroid set = bona fide + the candidates (NOT the kept anchors): these
    # are the directions we need to rank. Labels stay in the original numbering.
    cand_lbls = {name2lbl[c]: c for c in candidates}
    tr_names = {bf_full: full_names[bf_full], **cand_lbls}
    keep = np.isin(ytr_all, list(tr_names))
    ftr, ytr = ftr_all[keep], ytr_all[keep]
    te_names = unseen_names(cfg)

    acc: Dict[str, List[Dict[str, float]]] = {}
    for ck in cks:
        m = GeometricEncoder(dropout=cfg.geometric.dropout)
        m.load_state_dict(torch.load(ck, map_location="cpu")["model"])
        m.eval()
        with torch.no_grad():
            ztr = m(torch.from_numpy(ftr).float()).numpy()
            zte = m(torch.from_numpy(fte).float()).numpy()
        res = analyze_unseen_pais(ztr, ytr, tr_names, zte, yte, te_names,
                                  prototype_order=candidates,
                                  bf_key=full_names[bf_full],
                                  tau=cfg.attribution.tau)
        for u, r in res.items():
            acc.setdefault(u, []).append(dict(r["ranked_anchors"]))

    out = {u: {c: float(np.mean([d[c] for d in ds])) for c in candidates}
           for u, ds in acc.items()}
    # Scores that are constant across candidates would make argmax pick by dict
    # order, silently turning the xfpad arm into "always choose the first
    # candidate".
    flat = [tuple(round(s[c], 9) for c in candidates) for s in out.values()]
    if len(candidates) > 1 and all(len(set(f)) == 1 for f in flat):
        raise RuntimeError(
            f"[{scanner}] degenerate xfpad scores: every candidate ties for "
            f"every unseen PAI — the attribution is not discriminating.")
    return out


def _candidate_sizes(scanner: str, candidates: List[str]) -> Dict[str, int]:
    cfg = load_config(str(REPO / "configs" / f"{scanner}.yaml"))
    paths = read_split(split_path(cfg.paths.splits_dir, scanner, "train"))
    y = np.array(build_labels(paths, known_pairs(cfg)))
    name2lbl = {v: k for k, v in known_names(cfg).items()}
    return {c: int((y == name2lbl[c]).sum()) for c in candidates}


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank (paired, no scipy dependency)
# ---------------------------------------------------------------------------

def _wilcoxon(a: List[float], b: List[float]) -> float:
    d = [x - y for x, y in zip(a, b) if x != y]
    n = len(d)
    if n < 6:
        return float("nan")
    order = np.argsort([abs(x) for x in d])
    ranks = np.empty(n); ranks[order] = np.arange(1, n + 1)
    w = min(sum(r for r, x in zip(ranks, d) if x > 0),
            sum(r for r, x in zip(ranks, d) if x < 0))
    mu = n * (n + 1) / 4
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu) / sd
    from math import erf, sqrt
    return float(2 * 0.5 * (1 + erf(-abs(z) / sqrt(2))))


def _ranking_pairs(records: List[dict], subset=None):
    """Collect (predicted p_{u,c}, observed recovery(u,c)) over ALL candidates.

    This is the PRIMARY, single-pick-robust analysis: instead of asking only
    whether X-FPAD's top pick won, it asks whether X-FPAD's attribution ranks
    the *whole* candidate pool the way the observed recovery does. It is the
    forward analog of the ablation correlation (attribution predicts APCER
    shift), computed for
    free from cells trained anyway to evaluate the 'random' arm.
    """
    pred, obs = [], []
    for r in records:
        if subset and not subset(r):
            continue
        if "predicted_xfpad" not in r:
            continue
        for c, recov in r["recovery_per_candidate"].items():
            pred.append(r["predicted_xfpad"][c])
            obs.append(recov)
    return pred, obs


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3b analysis.")
    ap.add_argument("--geo-glob", default="geometric_{scanner}_reduced_[0-9].pth")
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    records: List[dict] = []
    for scanner, spec in PHASE4_SETUP.items():
        candidates = list(spec["candidates"])
        red = {bb: LOMO_OUT / f"phase3_{scanner}_{bb}_phase4_reduced.json" for bb in BACKBONES}
        add = {c: {bb: LOMO_OUT / f"phase3_{scanner}_{bb}_phase4_add_{_slug(c)}.json"
                   for bb in BACKBONES} for c in candidates}
        if not any(p.exists() for p in red.values()):
            print(f"[{scanner}] reduced baseline not computed yet — skipping")
            continue

        raw = _centroid_scores(scanner, candidates)
        xf = _xfpad_scores(scanner, candidates, args.geo_glob)
        if xf is None:
            print(f"[{scanner}] WARNING: no reduced-vocabulary encoder found; "
                  f"'xfpad' arm unavailable (train it first)")
        sizes = _candidate_sizes(scanner, candidates)

        for bb in BACKBONES:
            if not red[bb].exists():
                continue
            base = _per_pai(red[bb])
            rec = {c: _per_pai(add[c][bb]) for c in candidates if add[c][bb].exists()}
            if not rec:
                continue
            for u in base:
                recov = {c: base[u] - rec[c][u] for c in rec if u in rec[c]}
                if len(recov) < len(candidates):
                    continue          # need every candidate to compare strategies
                picks = {}
                if xf and u in xf:
                    picks["xfpad"] = max(recov, key=lambda c: xf[u][c])
                if u in raw:
                    picks["raw_nc"] = max(recov, key=lambda c: raw[u][c])
                    picks["most_diff"] = min(recov, key=lambda c: raw[u][c])
                if "xfpad" in picks:
                    tgt = sizes[picks["xfpad"]]
                    others = [c for c in recov if c != picks["xfpad"]]
                    picks["matched_size"] = min(others, key=lambda c: abs(sizes[c] - tgt))
                r = {"scanner": scanner, "backbone": bb, "unseen": u,
                     "recovery_per_candidate": recov,
                     "picks": picks,
                     "recovery": {s: recov[c] for s, c in picks.items()}}
                r["recovery"]["random"] = float(np.mean(list(recov.values())))
                if xf and u in xf:
                    r["predicted_xfpad"] = {c: xf[u][c] for c in recov}
                records.append(r)

    if not records:
        print("\nNo Phase 4 cells available yet. Run:  "
              "python scripts/lomo_driver.py --blocks phase4 --config-suffix _cons")
        return

    # ------------------------------------------------------------------
    # PRIMARY analysis: predicted-vs-observed candidate RANKING correlation
    # ------------------------------------------------------------------
    print("=" * 74)
    print("PRIMARY — does X-FPAD rank the candidates the way")
    print("observed recovery does?   Spearman(p_{u,c}, recovery(u,c))")
    print("=" * 74)
    pred, obs = _ranking_pairs(records)
    if pred:
        rho = _spearman(pred, obs)
        print(f"  OVERALL   rho = {rho:+.3f}   (n={len(pred)} (unseen,candidate) pairs)")
        for sc in PHASE4_SETUP:
            p, o = _ranking_pairs(records, lambda r, sc=sc: r["scanner"] == sc)
            if len(p) >= 3:
                print(f"    {sc:<10} rho = {_spearman(p, o):+.3f}  (n={len(p)})")
        for bb in BACKBONES:
            p, o = _ranking_pairs(records, lambda r, bb=bb: r["backbone"] == bb)
            if len(p) >= 3:
                print(f"    {bb:<12} rho = {_spearman(p, o):+.3f}  (n={len(p)})")
        print("  (rho > 0 => higher attribution predicts larger APCER recovery;")
        print("   robust to single-pick luck, unlike the top-1 table below.)")
    else:
        print("  no reduced-vocabulary encoder found -> 'xfpad' scores unavailable")
    print()

    print("=" * 74)
    print(f"COROLLARY — recovery of unseen APCER by selection strategy  "
          f"(n={len(records)} unseen x backbone)")
    print("=" * 74)
    print(f"{'strategy':<14} {'mean recovery':>15} {'median':>9} {'wins vs random':>16}")
    print("-" * 74)
    table = {}
    for s in STRATEGIES:
        vals = [r["recovery"][s] for r in records if s in r["recovery"]]
        if not vals:
            continue
        table[s] = vals
        wins = sum(v > r["recovery"]["random"]
                   for v, r in zip(vals, records) if s in r["recovery"])
        print(f"{s:<14} {np.mean(vals):>+15.2f} {np.median(vals):>+9.2f} "
              f"{wins:>10}/{len(vals)}")
    print("-" * 74)
    print("paired Wilcoxon (p-value):")
    for a, b in combinations([s for s in STRATEGIES if s in table], 2):
        if a == "xfpad" or b == "xfpad":
            print(f"   {a:<13} vs {b:<13} p = {_wilcoxon(table[a], table[b]):.4f}")

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"records": records,
                   "summary": {s: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                                   "n": len(v)} for s, v in table.items()}},
                  out.open("w"), indent=2)
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
