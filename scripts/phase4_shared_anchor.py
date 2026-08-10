"""Phase 3b read through SHARED ANCHORS on the reduced manifold.

Why this exists
---------------
`phase4_analysis.py` recommends a candidate by comparing the unseen PAI's
direction with the candidate's own direction on the reduced manifold. Both are
OUT OF VOCABULARY there -- neither has a trained prototype, so neither carries
the 2*pi/K angular separation the design guarantees -- and comparing two
unanchored objects is unstable: the candidate centroids sit a median 1.6 deg
apart on a given seed and their arrangement changes from seed to seed.

The trained prototypes are the one stable frame the reduced manifold has. So we
never compare a candidate with an unseen PAI directly. We attribute BOTH to the
kept materials' prototypes (Eq. 5) and recommend the candidate whose
attribution profile matches the unseen PAI's. Two out-of-vocabulary materials
are related transitively, through what the manifold does know.

Everything here is confined to the reduced vocabulary and to the 2-D manifold:

  * the encoder is `geometric_<scanner>_reduced_<seed>.pth`, trained only on the
    kept materials;
  * the prototypes are the kept materials' centroids, and nothing else;
  * candidates and unseen PAIs are only ever PROJECTED, never trained on;
  * no full-vocabulary encoder and no 1280-D feature ever enters a decision.

The 1280-D nearest-centroid baseline is computed too, as the external
reference in the source space -- it is reported alongside, never used to pick.

Estimator note
--------------
The attribution is the canonical one (xfpad/metrics/attribution.py): the cosine
softmax is taken PER SAMPLE and averaged over the cluster. Collapsing a cluster
to its centroid before the softmax is a different, much more peaked estimator
that inverts several picks; `--selftest` prints both so the difference stays
visible.

Usage
-----
    python scripts/phase4_shared_anchor.py --save-json outputs/point_e_shared_anchor.json
    python scripts/phase4_shared_anchor.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts._common import (  # noqa: E402
    bona_fide_label, known_names, known_pairs, unseen_names, unseen_pairs,
)
from scripts.lomo_driver import PHASE4_SETUP  # noqa: E402
from xfpad.config import load_config  # noqa: E402
from xfpad.data import build_labels  # noqa: E402
from xfpad.metrics import analyze_unseen_pais  # noqa: E402
from xfpad.utils import features_path, read_split, split_path  # noqa: E402

LOMO_OUT = REPO / "outputs" / "lomo"
PHASE4 = REPO / "outputs" / "phase4_analysis.json"
REDUCED_GLOB = "geometric_{scanner}_reduced_[0-9].pth"


# ---------------------------------------------------------------------------
# Reduced vocabulary
# ---------------------------------------------------------------------------

def vocabulary(scanner: str) -> Tuple[List[str], List[str]]:
    """(kept materials = prototypes, candidate materials = out of vocabulary).

    Derived from PHASE4_SETUP so this file cannot drift from the cells that
    were actually trained.
    """
    cfg = load_config(str(REPO / "configs" / f"{scanner}.yaml"))
    names = known_names(cfg)
    bf = bona_fide_label(names)
    spec = PHASE4_SETUP[scanner]
    cand = list(spec["candidates"])
    removed = set(spec["removed_for_reduced"])
    kept = [n for lbl, n in sorted(names.items())
            if lbl != bf and n not in cand
            and not any(sub.lower() in n.lower() for sub in removed)]
    return kept, cand


# ---------------------------------------------------------------------------
# Projection and attribution, reduced manifold only
# ---------------------------------------------------------------------------

def _load(scanner: str):
    cfg = load_config(str(REPO / "configs" / f"{scanner}.yaml"))
    names = known_names(cfg)
    bf = bona_fide_label(names)
    n2l = {v: k for k, v in names.items()}
    ftr = np.load(features_path(cfg.paths.features_dir, scanner, "train"))
    fte = np.load(features_path(cfg.paths.features_dir, scanner, "test"))
    ytr = np.array(build_labels(
        read_split(split_path(cfg.paths.splits_dir, scanner, "train")), known_pairs(cfg)))
    yte = np.array(build_labels(
        read_split(split_path(cfg.paths.splits_dir, scanner, "test")), unseen_pairs(cfg)))
    return cfg, names, bf, n2l, ftr, ytr, fte, yte


def _query_bucket(scanner, cand, ftr, ytr, fte, yte, n2l, names, bf):
    """Candidates (train samples) + unseen PAIs (test samples) in one namespace.

    Candidate samples are projected, never trained on: the practitioner owns the
    material, they have simply not committed it to the PAD's training set.
    """
    qf, ql, qn = [], [], {}
    nxt = 500
    for c in cand:
        m = ytr == n2l[c]
        assert m.sum() > 0, f"no training samples for candidate {c}"
        qf.append(ftr[m]); ql.append(np.full(m.sum(), nxt)); qn[nxt] = c; nxt += 1
    cfg = load_config(str(REPO / "configs" / f"{scanner}.yaml"))
    for lbl, n in unseen_names(cfg).items():
        if n == names[bf]:
            continue
        m = yte == int(lbl)
        if m.any():
            qf.append(fte[m]); ql.append(np.full(m.sum(), nxt)); qn[nxt] = n; nxt += 1
    return np.concatenate(qf), np.concatenate(ql), qn


def profiles_per_seed(scanner: str, centroid_softmax: bool = False):
    """[{name: p over kept prototypes}] -- one dict per reduced-encoder seed.

    `centroid_softmax=True` selects the NON-canonical variant (softmax of the
    centroid direction) purely so --selftest can contrast the two.
    """
    import torch
    from xfpad.models import GeometricEncoder

    kept, cand = vocabulary(scanner)
    cfg, names, bf, n2l, ftr, ytr, fte, yte = _load(scanner)
    qf, ql, qn = _query_bucket(scanner, cand, ftr, ytr, fte, yte, n2l, names, bf)

    proto_lbls = {bf: names[bf], **{n2l[k]: k for k in kept}}
    # GUARD: the prototype set is the reduced vocabulary and nothing else.
    assert set(proto_lbls.values()) == {names[bf], *kept}, "prototypes leaked"
    assert not (set(proto_lbls.values()) & set(cand)), "a candidate became a prototype"
    km = np.isin(ytr, list(proto_lbls))

    cks = sorted(glob.glob(str(REPO / "checkpoints" / REDUCED_GLOB.format(scanner=scanner))))
    assert cks, f"no reduced encoders for {scanner}"

    out = []
    for ck in cks:
        meta = torch.load(ck, map_location="cpu")
        # GUARD: this really is a reduced-vocabulary encoder.
        assert meta.get("K") == len(kept), (
            f"{Path(ck).name}: K={meta.get('K')} but the reduced vocabulary has "
            f"{len(kept)} materials -- wrong checkpoint family")
        mdl = GeometricEncoder(dropout=cfg.geometric.dropout)
        mdl.load_state_dict(meta["model"]); mdl.eval()
        with torch.no_grad():
            ztr = mdl(torch.from_numpy(ftr[km]).float()).numpy()
            zq = mdl(torch.from_numpy(qf).float()).numpy()
        assert ztr.shape[1] == 2 and zq.shape[1] == 2, "projection is not 2-D"

        if centroid_softmax:
            bfc = np.median(ztr[ytr[km] == bf], axis=0)
            unit = lambda v: v / max(np.linalg.norm(v), 1e-12)
            P = np.stack([unit(np.median(ztr[ytr[km] == n2l[k]], axis=0) - bfc) for k in kept])
            d = {}
            for lbl, nm in qn.items():
                v = unit(np.median(zq[ql == lbl], axis=0) - bfc)
                s = cfg.attribution.tau * (P @ v)
                e = np.exp(s - s.max()); d[nm] = e / e.sum()
            out.append(d)
        else:
            res = analyze_unseen_pais(ztr, ytr[km], proto_lbls, zq, ql, qn,
                                      prototype_order=kept, bf_key=names[bf],
                                      tau=cfg.attribution.tau)
            out.append({n: np.array([dict(r["ranked_anchors"])[k] for k in kept])
                        for n, r in res.items()})
    return out, kept, cand


def raw_nearest_centroid(scanner: str) -> Dict[str, str]:
    """External baseline: nearest candidate centroid in the 1280-D space.

    Reported for comparison only. It is deliberately NOT part of any decision
    path in this file.
    """
    kept, cand = vocabulary(scanner)
    cfg, names, bf, n2l, ftr, ytr, fte, yte = _load(scanner)
    qf, ql, qn = _query_bucket(scanner, cand, ftr, ytr, fte, yte, n2l, names, bf)
    cand_lbls = {bf: names[bf], **{n2l[c]: c for c in cand}}
    cm = np.isin(ytr, list(cand_lbls))
    un = {l: n for l, n in qn.items() if n not in cand}
    qm = np.isin(ql, list(un))
    res = analyze_unseen_pais(ftr[cm], ytr[cm], cand_lbls, qf[qm], ql[qm], un,
                              prototype_order=cand, bf_key=names[bf],
                              tau=cfg.attribution.tau)
    return {n: r["ranked_anchors"][0][0] for n, r in res.items()}


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def _cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def recommend(seeds, kept, cand, similarity="cosine"):
    """{unseen: (winner, votes, n_seeds, median top-2 margin)} by per-seed vote."""
    unseen = [n for n in seeds[0] if n not in cand]
    out = {}
    for u in unseen:
        picks, margins = [], []
        for s in seeds:
            if similarity == "cosine":
                sc = {c: _cos(s[u], s[c]) for c in cand}
            elif similarity == "l1":
                sc = {c: -float(np.abs(s[u] - s[c]).sum()) for c in cand}
            elif similarity == "argmax":            # shared top prototype, ties by cosine
                top = int(np.argmax(s[u]))
                sc = {c: (1.0 if int(np.argmax(s[c])) == top else 0.0) + 1e-3 * _cos(s[u], s[c])
                      for c in cand}
            else:
                raise ValueError(similarity)
            v = sorted(sc.values(), reverse=True)
            picks.append(max(sc, key=sc.get)); margins.append(v[0] - v[1])
        w, nw = Counter(picks).most_common(1)[0]
        out[u] = (w, nw, len(seeds), float(np.median(margins)))
    return out


# ---------------------------------------------------------------------------
# Scoring against the trained Phase 4 cells
# ---------------------------------------------------------------------------

def sign_test(d) -> Tuple[int, int, float]:
    nz = [x for x in d if x != 0]
    m = len(nz); k = sum(1 for x in nz if x > 0)
    if m == 0:
        return k, m, 1.0
    up = sum(comb(m, i) for i in range(k, m + 1)) / 2 ** m
    lo = sum(comb(m, i) for i in range(0, k + 1)) / 2 ** m
    return k, m, min(1.0, 2 * min(up, lo))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scanners", nargs="+", default=["greenbit", "dermalog"])
    ap.add_argument("--similarity", default="cosine", choices=["cosine", "l1", "argmax"])
    ap.add_argument("--selftest", action="store_true",
                    help="Contrast the canonical estimator with the centroid-softmax "
                         "variant and exit.")
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    if args.selftest:
        for sc in args.scanners:
            can, kept, cand = profiles_per_seed(sc, centroid_softmax=False)
            cen, _, _ = profiles_per_seed(sc, centroid_softmax=True)
            print(f"\n=== {sc} — canonical (mean of softmaxes) vs centroid-then-softmax ===")
            for est, tag in [(can, "canonical"), (cen, "centroid")]:
                r = recommend(est, kept, cand, args.similarity)
                for u, (w, nw, n, mg) in r.items():
                    print(f"  {tag:<10}{u:<26}{w:<14}{nw}/{n}  margine {mg:.4f}")
        return

    records = json.load(PHASE4.open())["records"]
    picks: Dict[Tuple[str, str], Tuple] = {}
    raw: Dict[Tuple[str, str], str] = {}
    for sc in args.scanners:
        seeds, kept, cand = profiles_per_seed(sc)
        print("=" * 96)
        print(f"{sc.upper()}  prototipi (vocabolario ridotto) = {kept}")
        print(f"{'':<10}out-of-vocabulary candidates      = {cand}")
        print("=" * 96)
        print("  mean attribution profile over the trained prototypes:")
        for n in list(cand) + [x for x in seeds[0] if x not in cand]:
            P = np.mean([s[n] for s in seeds], axis=0)
            tag = "CAND" if n in cand else "    "
            print(f"    {tag} {n:<26}" + "  ".join(f"{k}={v:.3f}" for k, v in zip(kept, P)))
        rec = recommend(seeds, kept, cand, args.similarity)
        rawp = raw_nearest_centroid(sc)
        print("\n  recommendation (one vote per seed):")
        for u, (w, nw, n, mg) in rec.items():
            picks[(sc, u)] = (w, nw, n, mg)
            raw[(sc, u)] = rawp[u]
            print(f"    {u:<26}{w:<14}{nw}/{n} seed   margine mediano {mg:.4f}")
        print()

    # cluster level: one unit per (scanner, unseen PAI), backbones averaged
    cl = defaultdict(lambda: defaultdict(list))
    for r in records:
        k = (r["scanner"], r["unseen"])
        if k not in picks:
            continue
        rc = r["recovery_per_candidate"]
        cl[k]["shared"].append(rc[picks[k][0]])
        cl[k]["raw_nc"].append(rc[raw[k]])
        for a in ("xfpad", "random", "matched_size", "most_diff"):
            if a in r["recovery"]:
                cl[k][a].append(r["recovery"][a])
        cl[k]["oracle"].append(max(rc.values()))

    units = sorted(cl)
    COLS = ["shared", "xfpad", "random", "matched_size", "most_diff", "raw_nc", "oracle"]
    M = {a: [] for a in COLS}
    print("=" * 132)
    print("RECOVERY per cluster unit (APCER points) — 'shared' and 'xfpad' use only "
          "the reduced 2-D manifold; 'raw_nc' is the 1280-D reference")
    print("=" * 132)
    print(f"{'scanner':<10}{'unseen PAI':<26}" + "".join(f"{a:>14}" for a in COLS))
    print("-" * 132)
    for k in units:
        line = f"{k[0]:<10}{k[1]:<26}"
        for a in COLS:
            v = float(np.mean(cl[k][a])) if cl[k][a] else float("nan")
            M[a].append(v); line += f"{v:>+14.2f}"
        print(line)
    print("-" * 132)
    for lab, fn in (("MEAN", np.mean), ("MEDIAN", np.median)):
        print(f"{lab:<36}" + "".join(f"{fn(M[a]):>+14.2f}" for a in COLS))
    print(f"{'% ORACLE':<36}{100 * np.mean(M['shared']) / np.mean(M['oracle']):>13.1f}%"
          + f"{100 * np.mean(M['xfpad']) / np.mean(M['oracle']):>13.1f}%"
          + f"{100 * np.mean(M['random']) / np.mean(M['oracle']):>13.1f}%")

    rng = np.random.default_rng(0)
    n = len(units)
    print("\n" + "=" * 132)
    print(f"'shared' against the alternatives, paired over n={n} cluster units")
    print("=" * 132)
    print(f"{'comparison':<34}{'mean diff':>13}{'bootstrap 95% CI':>24}{'wins':>10}{'sign p':>12}")
    print("-" * 132)
    stats = {}
    x = np.array(M["shared"])
    for a in ("random", "matched_size", "most_diff", "xfpad", "raw_nc"):
        d = x - np.array(M[a])
        bs = np.array([rng.choice(d, size=n, replace=True).mean() for _ in range(20000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        k, m, p = sign_test(d)
        stats[a] = {"delta": float(d.mean()), "ci": [float(lo), float(hi)],
                    "wins": k, "n_nonzero": m, "sign_p": float(p)}
        print(f"{'shared - ' + a:<34}{d.mean():>+13.2f}"
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>24}{f'{k}/{m}':>10}{p:>12.4f}")

    # top-1 at CLUSTER level: the candidate with the best backbone-averaged
    # recovery, which is the convention of the projection control, so the two
    # are directly comparable. (The
    # 'oracle' column above averages the per-backbone maxima instead, which is
    # optimistic and not a strategy any picker could realise.)
    per_cand = defaultdict(lambda: defaultdict(list))
    for r in records:
        k = (r["scanner"], r["unseen"])
        if k in picks:
            for c, v in r["recovery_per_candidate"].items():
                per_cand[k][c].append(v)
    top1 = {}
    for arm, chooser in (("shared", lambda k: picks[k][0]),
                         ("raw_nc", lambda k: raw[k])):
        top1[arm] = sum(1 for k in units
                        if chooser(k) == max(per_cand[k],
                                             key=lambda c: np.mean(per_cand[k][c])))
    print("\ntop-1 at cluster level (candidate with the best mean recovery):")
    for arm, h in top1.items():
        print(f"  {arm:<10}{h}/{n}")

    if args.save_json:
        out = {"units": [{"scanner": k[0], "unseen": k[1],
                          "pick": picks[k][0], "votes": picks[k][1],
                          "seeds": picks[k][2], "margin": picks[k][3],
                          "raw_nc_pick": raw[k],
                          **{a: float(np.mean(cl[k][a])) for a in COLS if cl[k][a]}}
                         for k in units],
               "means": {a: float(np.mean(M[a])) for a in COLS},
               "paired_vs_shared": stats,
               "similarity": args.similarity, "n_units": n}
        p = Path(args.save_json)
        tmp = p.with_suffix(".json.tmp")
        with tmp.open("w") as fh:
            json.dump(out, fh, indent=2)
        tmp.replace(p)
        print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
