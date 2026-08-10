"""How much does the Phase 2 reading depend on the run?

The published attribution is the average over the ten runs of g_psi, and it is
that average which decides the (material, unseen PAI) pairs above the
p_{u,k} >= 0.30 threshold, i.e. the ones Phase 3a then tests by removing the
material. If one of those decisions flipped from run to run, the average would
be hiding it.

The question is answered on the single runs, not on the averages: the per-run
attributions are on disk, so the pairs each run selects are counted directly
rather than inferred from the dispersion of the mean.

Usage
-----
    python scripts/phase2_run_stability.py
    python scripts/phase2_run_stability.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCANNERS = ("greenbit", "dermalog")

sys.path.insert(0, str(REPO))
from scripts._protocol import P_THRESHOLD as THR  # noqa: E402


def load(scanner: str) -> dict:
    p = REPO / "outputs" / f"phase2_{scanner}_intra.json"
    assert p.exists(), f"{p} missing"
    return json.loads(p.read_text(encoding="utf-8"))


def dominant(d: dict) -> dict[str, set[str]]:
    """{PAI: set of dominant anchors seen across the runs}"""
    out: dict[str, set[str]] = {}
    for run in d["per_seed"]:
        for pai, node in run["attribution"].items():
            out.setdefault(pai, set()).add(node["ranked"][0][0])
    return out


def selected(d: dict) -> tuple[set, list[set], Counter]:
    """(PAI, material) pairs above threshold: from the mean, per run, and counts."""
    agg = d["aggregated"]["attribution"]
    mean = {(pai, m) for pai, node in agg.items()
            for m, w in node["mean"].items() if w >= THR}
    per_run, counts = [], Counter()
    for run in d["per_seed"]:
        sel = {(pai, m) for pai, node in run["attribution"].items()
               for m, w in node["weights"].items() if w >= THR}
        per_run.append(sel)
        counts.update(sel)
    return mean, per_run, counts


def report() -> dict:
    acc: dict = {}
    for sc in SCANNERS:
        d = load(sc)
        n_runs = len(d["per_seed"])
        dom = dominant(d)
        mean, per_run, counts = selected(d)
        agreeing = sum(1 for s in per_run if s == mean)
        extra = {u for s in per_run for u in s} - mean
        below = sorted(u for u in mean if counts[u] < n_runs)
        acc[sc] = dict(n_runs=n_runs, dominant_unique=max(len(v) for v in dom.values()),
                       n_pai=len(dom), n_mean=len(mean), agreeing=agreeing,
                       extra=extra, below=[(u, counts[u]) for u in below],
                       weights={u: d["aggregated"]["attribution"][u[0]]["mean"][u[1]]
                                for u in below})
        print(f"  [{sc}] {n_runs} runs, {len(dom)} unseen PAIs")
        print(f"    dominant anchor: {'the same in every run' if acc[sc]['dominant_unique'] == 1 else 'CHANGES'}")
        print(f"    pairs above {THR:.2f} from the mean: {len(mean)}; "
              f"runs reproducing them exactly: {agreeing}/{n_runs}")
        for u, c in acc[sc]["below"]:
            print(f"      {c}/{n_runs}  {u[1]} on {u[0]}  (mean weight {acc[sc]['weights'][u]:.2f})")
        print(f"    pairs a run selects and the mean does not: {len(extra) or 'none'}")
    return acc


def selftest() -> None:
    acc = report()
    tot_mean = sum(a["n_mean"] for a in acc.values())
    assert tot_mean == 11, f"{tot_mean} predicted units, the text says 11"
    for sc, a in acc.items():
        assert a["n_runs"] == 10, a["n_runs"]
        assert a["dominant_unique"] == 1, f"{sc}: the dominant anchor changes between runs"
        assert not a["extra"], f"{sc}: a run selects pairs the mean does not have"
    assert acc["dermalog"]["agreeing"] == 10, acc["dermalog"]["agreeing"]
    assert acc["greenbit"]["agreeing"] == 8, acc["greenbit"]["agreeing"]
    # the two dissenting runs each lose one pair, and it is one of the two at 0.32
    gb = acc["greenbit"]
    assert len(gb["below"]) == 2 and all(c == 9 for _u, c in gb["below"]), gb["below"]
    assert all(abs(w - 0.32) < 0.005 for w in gb["weights"].values()), gb["weights"]
    names = {(u[0], u[1]) for u, _c in gb["below"]}
    assert names == {("Elmer's Glue Consensual", "Latex"),
                     ("Mix1 ScreenSpoof", "Body Double")}, names
    print("  selftest ok --- the statements in the text hold on the single runs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    report()


if __name__ == "__main__":
    main()
