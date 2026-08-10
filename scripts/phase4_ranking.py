"""Phase 3b rank correlation: the forward analogue of the Phase 3a rho.

Phase 3a shows that the attribution weight orders the effects observed when a
material is REMOVED. This asks the same question forward: does the manifold
order the candidates the way the recovery obtained by REINSTATING them does?

The strategy comparison of the main table is not enough for that: it depends on
which single candidate is picked, and at n = 10 units a lucky pick is not
distinguishable from a method that works. The rank correlation uses ALL the
(unseen PAI, candidate) pairs and never looks at the argmax.

Two differences from the namesake computation in phase4_analysis.py, which make
this the publishable number and that one not:

  * the score is the SHARED-ANCHOR one -- the similarity between attribution
    profiles over the retained prototypes -- the same quantity that produces the
    recovery reported in the main paper, not the direct comparison of the two
    out-of-vocabulary directions;
  * the unit is the CLUSTER (scanner, unseen PAI, candidate) with the audited
    systems averaged, not the per-system pair: the systems share data, splits
    and encoders, so counting them separately would be counting replicates.

Usage
-----
    python scripts/phase4_ranking.py
    python scripts/phase4_ranking.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.lomo_driver import AUDITED  # noqa: E402
from scripts.phase4_shared_anchor import _cos, profiles_per_seed  # noqa: E402

RECORDS = REPO / "outputs" / "phase4_analysis.json"
SCANNERS = ("greenbit", "dermalog")


def pairs() -> list[dict]:
    """One row per (scanner, unseen PAI, candidate), audited systems averaged."""
    recs = json.loads(RECORDS.read_text(encoding="utf-8"))["records"]
    rec: dict[tuple, list[float]] = defaultdict(list)
    for r in recs:
        for c, v in r["recovery_per_candidate"].items():
            rec[(r["scanner"], r["unseen"], c)].append(v)

    rows = []
    for scanner in SCANNERS:
        seeds, _kept, cand = profiles_per_seed(scanner)
        unseen = [n for n in seeds[0] if n not in cand]
        for u in unseen:
            for c in cand:
                key = (scanner, u, c)
                if key not in rec:
                    continue
                assert len(rec[key]) == len(AUDITED), \
                    f"{key}: expected {len(AUDITED)} audited systems, found {len(rec[key])}"
                rows.append({
                    "scanner": scanner, "unseen": u, "cand": c,
                    "score": float(np.mean([_cos(s[u], s[c]) for s in seeds])),
                    "recovery": float(np.mean(rec[key])),
                })
    return rows


def rho(rows: list[dict], scanner: str | None = None) -> tuple[float, float, int]:
    sel = [r for r in rows if scanner is None or r["scanner"] == scanner]
    r, p = stats.spearmanr([x["score"] for x in sel], [x["recovery"] for x in sel])
    return float(r), float(p), len(sel)


def selftest() -> None:
    rows = pairs()
    # 6 unseen PAIs x 3 candidates + 4 x 2
    assert len(rows) == 26, len(rows)
    assert sum(1 for r in rows if r["scanner"] == "greenbit") == 18
    assert sum(1 for r in rows if r["scanner"] == "dermalog") == 8

    r, p, n = rho(rows)
    assert abs(r - 0.565) < 0.005, r
    assert p < 0.01, p
    # it must be the same size as the Phase 3a rho (+0.523 over the four
    # systems): that is what the main paper claims by calling it the forward
    # counterpart
    assert abs(r - 0.523) < 0.05, r
    # positive on each sensor taken separately
    for sc in SCANNERS:
        rr, _, nn = rho(rows, sc)
        assert rr > 0, f"{sc}: rho = {rr}"
        assert nn in (18, 8)
    print(f"selftest ok — {n} cluster pairs, rho = {r:+.3f} (p = {p:.4f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    rows = pairs()
    if args.selftest:
        selftest()
        return

    r, p, n = rho(rows)
    print(f"OVERALL    rho = {r:+.3f}   p = {p:.4f}   (n={n} cluster pairs)")
    for sc in SCANNERS:
        rr, pp, nn = rho(rows, sc)
        print(f"  {sc:<9} rho = {rr:+.3f}   p = {pp:.4f}   (n={nn})")


if __name__ == "__main__":
    main()
