"""Does the reading of one unseen PAI depend on which others were in the batch?

A diagnostic is used one arrival at a time: an unseen PAI appears, the manifold
is asked which training material governs it, and the answer is acted on. The
question here is whether that answer survives a change in the company the PAI
keeps.

X-FPAD cannot fail this test by construction --- the encoder is frozen, the
prototypes come from the training clusters, and the attribution of a PAI reads
only its own samples --- so its row is the control that the measurement does
what it says. The post-hoc reductions have no such guarantee: t-SNE and UMAP
are refitted from scratch whenever the set of points changes, and a refit moves
the known materials too.

One unseen PAI is withdrawn at a time and the anchors of the remaining ones are
compared against the run with all of them present. Withdrawing one of several
keeps the sample size nearly unchanged, so a moved anchor cannot be blamed on
the density these methods depend on. The margin between the dominant anchor and
the runner-up is recorded as well: a flip between two weights that were tied is
a different matter from a flip that overturns a decisive reading.

Usage
-----
    python scripts/context_stability.py --save-json
    python scripts/context_stability.py --selftest
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
from xfpad.config import load_config  # noqa: E402
from xfpad.data import build_labels  # noqa: E402
from xfpad.metrics import analyze_unseen_pais  # noqa: E402
from xfpad.utils import (  # noqa: E402
    features_path, geometric_ckpt, read_split, split_path)

SCANNERS = ("greenbit", "dermalog")
# 'pca' is the reduction of Table S7, fitted jointly on the training and unseen
# samples so that every representation receives one protocol. That protocol is
# obligatory for t-SNE and UMAP and merely available for PCA, which can be
# fitted on the training set alone and applied to new points; 'pca_train' is
# that second reading, and the distance between the two is the cost of the
# uniform treatment rather than a property of PCA.
REPRESENTATIONS = ("raw", "pca", "pca_train", "tsne", "umap",
                   "xfpad_radialonly", "xfpad_cosface", "xfpad_arcface", "xfpad")
# the trained variants read through their own frozen encoder; the projection
# path is the one of 'xfpad' with a different checkpoint
VARIANT_CKPT = {"xfpad_radialonly": "radialonly", "xfpad_cosface": "cosface",
                "xfpad_arcface": "arcface"}
TAU = 5.0
OUT = REPO / "outputs" / "context_stability.json"


def _reading(rep: str, cfg, ftr, ztr_labels, fte, yte, mask,
             tr_names, te_names, order, bf, ckpt) -> Dict[str, tuple]:
    """{unseen PAI: (dominant anchor, margin over the runner-up)}"""
    if rep == "pca_train":
        from sklearn.decomposition import PCA
        p = PCA(n_components=2, random_state=cfg.seed).fit(ftr)
        z_train, z_test = p.transform(ftr), p.transform(fte[mask])
    elif rep in VARIANT_CKPT:
        alt = ckpt.parent / ckpt.name.replace(
            f"_{cfg.seed}.pth", f"_{VARIANT_CKPT[rep]}_{cfg.seed}.pth")
        z_train, z_test = _reduce("xfpad", ftr, fte[mask], cfg.seed, alt,
                                  cfg.geometric.dropout)
    else:
        z_train, z_test = _reduce(rep, ftr, fte[mask], cfg.seed, ckpt,
                                  cfg.geometric.dropout)
    res = analyze_unseen_pais(
        features_train=np.asarray(z_train), labels_train=ztr_labels,
        train_names=tr_names, features_unseen=np.asarray(z_test),
        labels_unseen=yte[mask], unseen_names=te_names,
        prototype_order=order, bf_key=tr_names[bf], tau=TAU)
    out = {}
    for pai, r in res.items():
        ranked = r["ranked_anchors"]
        margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
        out[pai] = (ranked[0][0], float(margin))
    return out


def measure(only: List[str] | None = None) -> dict:
    reps = tuple(only) if only else REPRESENTATIONS
    results: Dict[str, dict] = {r: {"changed": 0, "total": 0, "flips": [],
                                    "margin_flipped": [], "margin_held": []}
                                for r in reps}
    for scanner in SCANNERS:
        cfg = load_config(f"configs/{scanner}.yaml")
        ftr = np.load(features_path(cfg.paths.features_dir, cfg.scanner, "train"))
        fte = np.load(features_path(cfg.paths.features_dir, cfg.scanner, "test"))
        ytr = np.asarray(build_labels(read_split(split_path(
            cfg.paths.splits_dir, cfg.scanner, "train")), known_pairs(cfg)))
        yte = np.asarray(build_labels(read_split(split_path(
            cfg.paths.splits_dir, cfg.scanner, "test")), unseen_pairs(cfg)))
        tr_names, te_names = known_names(cfg), unseen_names(cfg)
        bf = bona_fide_label(tr_names)
        order = [tr_names[l] for l in sorted(tr_names) if l != bf]
        ckpt = geometric_ckpt(cfg.paths.checkpoints, cfg.scanner, cfg.seed)
        allmask = np.ones(len(yte), dtype=bool)

        for rep in reps:
            full = _reading(rep, cfg, ftr, ytr, fte, yte, allmask,
                            tr_names, te_names, order, bf, ckpt)
            # one margin per unseen PAI, from the reading with all present:
            # the per-comparison lists below repeat each PAI several times and
            # would weight the sensors by how many PAIs they contribute
            results[rep].setdefault("margins", []).extend(
                m for _anchor, m in full.values())
            # only labels the attribution actually produces: the bona fide
            # class of the unseen split is not a PAI and carries no anchor
            pai_labels = [l for l, n in te_names.items() if n in full]
            for drop in pai_labels:
                sub = _reading(rep, cfg, ftr, ytr, fte, yte, yte != drop,
                               tr_names, te_names, order, bf, ckpt)
                for lbl in pai_labels:
                    if lbl == drop:
                        continue
                    pai = te_names[lbl]
                    anchor, margin = full[pai]
                    results[rep]["total"] += 1
                    if sub[pai][0] != anchor:
                        results[rep]["changed"] += 1
                        results[rep]["margin_flipped"].append(margin)
                        results[rep]["flips"].append(
                            {"scanner": scanner, "withdrawn": te_names[drop],
                             "pai": pai, "from": anchor, "to": sub[pai][0],
                             "margin": round(margin, 3)})
                    else:
                        results[rep]["margin_held"].append(margin)
            print(f"  [{scanner}] {rep}: {results[rep]['changed']}"
                  f"/{results[rep]['total']}", flush=True)
    for r in results.values():
        for k in ("margin_flipped", "margin_held"):
            r[k.replace("margin", "median")] = (
                float(np.median(r[k])) if r[k] else None)
        r["median_margin"] = float(np.median(r["margins"]))
    return results


def report(res: dict) -> None:
    print("\n" + "=" * 78)
    print("DOES THE ANCHOR SURVIVE A CHANGE OF BATCH?")
    print("=" * 78)
    print(f"{'representation':<20} {'moved':>10} {'share':>8} "
          f"{'median margin':>15} {'moved':>7} {'held':>7}")
    print("-" * 78)
    for rep in REPRESENTATIONS:
        if rep not in res:
            continue
        r = res[rep]
        mf = "--" if r["median_flipped"] is None else f"{r['median_flipped']:.2f}"
        mh = "--" if r["median_held"] is None else f"{r['median_held']:.2f}"
        print(f"{rep:<20} {r['changed']:>4}/{r['total']:<5} "
              f"{100 * r['changed'] / r['total']:>7.1f}% "
              f"{r['median_margin']:>15.2f} {mf:>7} {mh:>7}")
    print("-" * 78)
    for rep in REPRESENTATIONS:
        for f in res.get(rep, {}).get("flips", [])[:4]:
            print(f"  {rep:<8} {f['scanner']:<9} withdrawing {f['withdrawn']:<24} "
                  f"{f['pai']:<24} {f['from']} -> {f['to']}  (margin {f['margin']})")


def selftest() -> None:
    assert OUT.exists(), f"{OUT} missing; run with --save-json"
    res = json.loads(OUT.read_text(encoding="utf-8"))
    # the control: a frozen encoder reading a PAI's own samples cannot depend
    # on the company it keeps
    assert res["xfpad"]["changed"] == 0, res["xfpad"]["flips"][:3]
    # nor can the 1280-D space, which is not fitted at all, nor PCA once it is
    # fitted on the training set and applied to the new points. Invariance
    # therefore does not separate the representations on its own.
    for rep in ("raw", "pca_train"):
        assert res[rep]["changed"] == 0, (rep, res[rep]["flips"][:3])
    # the two that must be refitted whenever the sample changes do move
    assert res["umap"]["changed"] > res["tsne"]["changed"] > 0, res
    # PCA moves only under the joint protocol, and only between anchors that
    # were tied: the movement is the cost of the uniform treatment
    assert res["pca"]["changed"] > 0 and res["pca"]["median_flipped"] < 0.01, res["pca"]
    # what does separate them is decisiveness: the dominant anchor of the
    # manifold leads the runner-up by an order of magnitude more than any
    # two-dimensional reduction manages
    # every trained encoder is frozen, so none of them can depend on the batch
    for rep in ("xfpad_radialonly", "xfpad_cosface", "xfpad_arcface"):
        assert res[rep]["changed"] == 0, (rep, res[rep]["flips"][:3])
    marg = {r: res[r]["median_margin"] for r in res}
    # Decisiveness separates the trained geometries from the two-dimensional
    # reductions, and not X-FPAD from the rest: the free-prototype encoders are
    # as decisive, ArcFace marginally more so. What is left to separate them is
    # the rank correlation and the identity of the sectors across retrainings.
    for rep in ("pca", "pca_train", "tsne", "umap"):
        assert marg[rep] <= 0.07, (rep, marg[rep])
    for rep in ("xfpad", "xfpad_cosface", "xfpad_arcface"):
        assert marg[rep] >= 0.30, (rep, marg[rep])
    # without the angular term there is no dominant anchor to speak of
    assert marg["xfpad_radialonly"] < 0.01, marg["xfpad_radialonly"]
    assert len({res[r]["total"] for r in res}) == 1, "unequal comparisons"
    print(f"  selftest ok — X-FPAD, raw and train-fitted PCA hold every one of "
          f"{res['xfpad']['total']} comparisons; t-SNE moves "
          f"{res['tsne']['changed']} and UMAP {res['umap']['changed']}; "
          f"median margin {marg['xfpad']:.2f} against at most "
          f"{max(marg[r] for r in ('pca', 'pca_train', 'tsne', 'umap')):.2f} "
          f"for a two-dimensional reduction")


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch dependence of the anchor.")
    ap.add_argument("--save-json", nargs="?", const=str(OUT), default=None)
    ap.add_argument("--only", nargs="+", default=None,
                    help="measure a subset; the saved file keeps the rest, so "
                         "a cheap representation can be added without refitting "
                         "t-SNE and UMAP.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    res = measure(args.only)
    if args.save_json:
        p = Path(args.save_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        if args.only and p.exists():
            merged = json.loads(p.read_text(encoding="utf-8"))
            merged.update(res)
            res = merged
        json.dump(res, p.open("w"), indent=2)
    report(res)
    if args.save_json:
        print(f"\nsaved -> {args.save_json}")


if __name__ == "__main__":
    main()
