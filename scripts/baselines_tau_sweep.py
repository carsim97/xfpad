"""Each diagnostic baseline at the inverse temperature that suits it best.

Table S7 compares representations through one estimator with one inverse
temperature, tau = 5. That temperature was fixed on the manifold, where the
cosine between centroid directions spreads over a wide range; in the 1280-D
embedding the same directions are closer to orthogonal, so an identical tau
produces a flatter softmax. Part of the entropy gap could therefore be the
temperature rather than the representation.

The objection is answered by giving it away: every alternative is swept over
tau and reported at the value that maximises its own rank correlation with the
ablation outcome, against X-FPAD at the tau the paper uses. Beating a rival at
the rival's best setting is the claim worth making.

Nothing is retrained. The reduction of each representation is computed once ---
t-SNE and UMAP are fitted jointly on the training and unseen samples, which is
the most favourable treatment available to methods that cannot embed new points
--- and only the softmax is recomputed per tau.

Usage
-----
    python scripts/baselines_tau_sweep.py
    python scripts/baselines_tau_sweep.py --save-json outputs/tau_sweep.json
    python scripts/baselines_tau_sweep.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts._common import (  # noqa: E402
    bona_fide_label, known_names, known_pairs, unseen_names, unseen_pairs)
from scripts.baselines_attribution import _reduce  # noqa: E402
from scripts.baselines_predictiveness import GEO_VARIANTS, clusters, score  # noqa: E402
from scripts.correlate_attr_apcer import attribution_matrix, shift_records  # noqa: E402
from xfpad.config import load_config  # noqa: E402
from xfpad.data import build_labels  # noqa: E402
from xfpad.metrics import analyze_unseen_pais  # noqa: E402
from xfpad.utils import features_path, geometric_ckpt, read_split, split_path  # noqa: E402

SCANNERS = ("greenbit", "dermalog")
# both protocols of each projection: a temperature that suits the joint fit
# need not suit the train-fitted reading, and the claim the sweep supports has
# to hold for either
POSTHOC = ("raw", "pca", "pca_train", "tsne", "tsne_train", "umap", "umap_train")
# wide enough on the right for the near-orthogonal 1280-D case, where a sharper
# softmax is what the representation needs to name anchors at all
TAUS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0)
PAPER_TAU = 5.0
SWEEP = REPO / "outputs" / "tau_sweep.json"


def _load(scanner: str):
    cfg = load_config(f"configs/{scanner}.yaml")
    ftr = np.load(features_path(cfg.paths.features_dir, cfg.scanner, "train"))
    fte = np.load(features_path(cfg.paths.features_dir, cfg.scanner, "test"))
    ytr = build_labels(read_split(split_path(cfg.paths.splits_dir, cfg.scanner,
                                             "train")), known_pairs(cfg))
    yte = build_labels(read_split(split_path(cfg.paths.splits_dir, cfg.scanner,
                                             "test")), unseen_pairs(cfg))
    return cfg, ftr, fte, ytr, yte


def _weights(res: dict) -> Dict[str, Dict[str, float]]:
    """{unseen PAI: {material: p}} from an analyze_unseen_pais result."""
    return {pai: {m: float(w) for m, w in r["ranked_anchors"]}
            for pai, r in res.items()}


def posthoc_attributions() -> Dict[str, Dict[float, Dict[str, dict]]]:
    """{representation: {tau: {scanner: {unseen: {material: p}}}}}"""
    out: Dict[str, Dict[float, Dict[str, dict]]] = {
        r: {t: {} for t in TAUS} for r in POSTHOC}
    for scanner in SCANNERS:
        cfg, ftr, fte, ytr, yte = _load(scanner)
        tr_names, te_names = known_names(cfg), unseen_names(cfg)
        bf = bona_fide_label(tr_names)
        order = [tr_names[l] for l in sorted(tr_names) if l != bf]
        for rep in POSTHOC:
            print(f"  [{scanner}] reducing '{rep}' ...", flush=True)
            ztr, zte = _reduce(rep, ftr, fte, cfg.seed,
                               geometric_ckpt(cfg.paths.checkpoints,
                                              cfg.scanner, cfg.seed),
                               cfg.geometric.dropout)
            for tau in TAUS:
                res = analyze_unseen_pais(
                    features_train=np.asarray(ztr), labels_train=np.asarray(ytr),
                    train_names=tr_names, features_unseen=np.asarray(zte),
                    labels_unseen=np.asarray(yte), unseen_names=te_names,
                    prototype_order=order, bf_key=tr_names[bf], tau=tau)
                out[rep][tau][scanner] = _weights(res)
    return out


def geo_attributions() -> Dict[str, Dict[float, Dict[str, dict]]]:
    out: Dict[str, Dict[float, Dict[str, dict]]] = {}
    for name, glob in GEO_VARIANTS.items():
        out[name] = {}
        for tau in TAUS:
            per = {}
            for scanner in SCANNERS:
                try:
                    per[scanner] = attribution_matrix(scanner, glob, tau)
                except FileNotFoundError:
                    pass
            out[name][tau] = per
        print(f"  swept '{name}'", flush=True)
    return out


def sweep() -> dict:
    records: List[dict] = []
    for sc in SCANNERS:
        records += shift_records(sc)
    assert records, "no LOMO shifts on disk"

    attrs = {**posthoc_attributions(), **geo_attributions()}
    out: Dict[str, dict] = {}
    for rep, per_tau in attrs.items():
        rows = {}
        for tau, per_scanner in per_tau.items():
            if not per_scanner:
                continue
            st = score(clusters(records, per_scanner))
            if st:
                rows[tau] = st
        if not rows:
            continue
        best = max(rows, key=lambda t: rows[t]["spearman"])
        # The dominant anchor of a cluster is not invariant in tau: the softmax
        # is monotone in the cosine per sample, but the weights are averaged
        # over the cluster afterwards and a mean does not preserve an argmax.
        # What matters is the range over which the set of dominant anchors is
        # unchanged, so the signature is kept per temperature.
        for tau, per_scanner in per_tau.items():
            if tau in rows:
                rows[tau]["argmax_key"] = repr(sorted(
                    (sc, u, max(w, key=w.get))
                    for sc, units in per_scanner.items()
                    for u, w in units.items()))
        keys = {rows[t]["argmax_key"] for t in rows}
        out[rep] = {"by_tau": {str(t): rows[t] for t in rows},
                    "best_tau": best, "best": rows[best],
                    "paper_tau": rows.get(PAPER_TAU),
                    "argmax_stable_over_tau": len(keys) == 1}
    return out


def plateau(rep: dict) -> List[float]:
    """Temperatures whose reading is identical to the one at PAPER_TAU."""
    ref = rep["by_tau"][str(PAPER_TAU)]
    return sorted(float(t) for t, s in rep["by_tau"].items()
                  if (s["directional_n"], s["directional_hits"],
                      s["argmax_key"]) == (ref["directional_n"],
                                           ref["directional_hits"],
                                           ref["argmax_key"]))


def report(res: dict) -> None:
    ref = res["xfpad"]["paper_tau"]
    print("\n" + "=" * 96)
    print("EACH REPRESENTATION AT ITS OWN BEST INVERSE TEMPERATURE")
    print("=" * 96)
    print(f"{'representation':<20} {'best tau':>9} {'rho':>8} {'anchors':>9} "
          f"{'signs':>8} {'selectivity':>12}   {'rho at tau=5':>13}")
    print("-" * 96)
    for rep in ("raw", "pca", "tsne", "umap", "xfpad_radialonly",
                "xfpad_cosface", "xfpad_arcface", "xfpad"):
        if rep not in res:
            continue
        r, b = res[rep], res[rep]["best"]
        at5 = r["paper_tau"]["spearman"] if r["paper_tau"] else float("nan")
        print(f"{rep:<20} {r['best_tau']:>9.1f} {b['spearman']:>+8.3f} "
              f"{b['directional_n']:>9} {b['directional_hits']:>8} "
              f"{100 * b['marginal_inside_floor']:>11.1f}%   {at5:>+13.3f}")
    print("-" * 96)
    print(f"X-FPAD at the tau the paper uses: rho = {ref['spearman']:+.3f}, "
          f"{ref['directional_hits']}/{ref['directional_n']} anchors, "
          f"{100 * ref['marginal_inside_floor']:.1f}% selectivity")
    beaten = [r for r in res if r != "xfpad"
              and res[r]["best"]["spearman"] >= ref["spearman"]]
    print(f"alternatives matching or beating it at their own best tau: "
          f"{beaten if beaten else 'none'}")
    unstable = [r for r in res if not res[r]["argmax_stable_over_tau"]]
    print(f"representations whose dominant anchor moves with tau: "
          f"{unstable if unstable else 'none'}")


def selftest() -> None:
    assert SWEEP.exists(), f"{SWEEP} missing; run this script with --save-json"
    res = json.loads(SWEEP.read_text(encoding="utf-8"))
    ref = res["xfpad"]["paper_tau"]

    # the claim the sweep exists to support: giving every alternative the
    # temperature that suits it best is not enough to reach the manifold
    for rep in res:
        if rep == "xfpad":
            continue
        assert res[rep]["best"]["spearman"] < ref["spearman"], \
            (rep, res[rep]["best_tau"], res[rep]["best"]["spearman"])

    # tau = 5 is not the manifold's own optimum, and adopting the optimum
    # would cost anchors rather than gain them
    assert res["xfpad"]["best_tau"] < PAPER_TAU, res["xfpad"]["best_tau"]
    assert res["xfpad"]["best"]["spearman"] - ref["spearman"] < 0.01, res["xfpad"]
    assert res["xfpad"]["best"]["directional_n"] < ref["directional_n"], res["xfpad"]

    # the published reading holds from tau = 5 to the top of the range
    pl = plateau(res["xfpad"])
    assert pl == [t for t in TAUS if t >= PAPER_TAU], pl
    assert (ref["directional_n"], ref["directional_hits"]) == (11, 11), ref

    # the anchor count is not what separates the representations: at its own
    # best temperature t-SNE names more anchors than the manifold does
    tsne = res["tsne"]["by_tau"]
    assert max(s["directional_n"] for s in tsne.values()) > ref["directional_n"]

    print(f"  selftest ok — no alternative reaches rho = {ref['spearman']:+.3f} "
          f"at any tau in {TAUS[0]}..{TAUS[-1]}; the reading is unchanged over "
          f"tau in {pl[0]:.0f}..{pl[-1]:.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep the attribution temperature.")
    ap.add_argument("--save-json", nargs="?", const=str(SWEEP), default=None,
                    help=f"write the sweep (default target: {SWEEP.name}).")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    res = sweep()
    report(res)
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, out.open("w"), indent=2)
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
