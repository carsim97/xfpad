"""LaTeX body of the Phase 3b table.

Reporting RECOVERY alone -- a difference -- hides where it starts from: "18.4
points recovered" does not distinguish a detector falling from 25% to 6% from
one falling from 80% to 61%. Across the 10 unseen PAIs the reduced-vocabulary
APCER spans 2.3% to 91.9%, so an average of recoveries is dominated by the units
that had room.

The table therefore reports the ABSOLUTE APCER after reinstatement, framed by
two reference rows: the reduced vocabulary with nothing added, and the best
candidate chosen with hindsight. Ordered by decreasing APCER it reads as a
scale, from doing nothing to making the perfect choice. Recovery stays in the
prose, where the argument needs it.

The three count columns are the paired comparison against the recommendation,
with ties shown rather than dropped: every row sums to 10.

Usage
-----
    python scripts/emit_phase4_table.py
    python scripts/emit_phase4_table.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LOMO = REPO / "outputs" / "lomo"
SHARED = REPO / "outputs" / "point_e_shared_anchor.json"
sys.path.insert(0, str(REPO))
from scripts._protocol import AUDITED  # noqa: E402

# (JSON key, table label)
ARMS = [("most_diff", "most dissimilar candidate"),
        ("random", "random"),
        ("shared", r"\textbf{X-FPAD, shared anchors}"),
        ("raw_nc", "nearest centroid, 1280-D"),
        ("oracle", "oracle")]


def reduced_apcer() -> dict:
    """Reduced-vocabulary APCER per (scanner, PAI), audited systems averaged."""
    acc: dict[tuple, list[float]] = defaultdict(list)
    for scanner in ("greenbit", "dermalog"):
        for bb in AUDITED:
            p = LOMO / f"phase3_{scanner}_{bb}_phase4_reduced.json"
            if not p.exists():
                continue
            for u, v in json.loads(p.read_text(encoding="utf-8"))[
                    "aggregated"]["per_pai"].items():
                acc[(scanner, u)].append(float(v["mean"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def rows() -> tuple[float, list[dict]]:
    units = json.loads(SHARED.read_text(encoding="utf-8"))["units"]
    red = reduced_apcer()
    base = float(np.mean([red[(u["scanner"], u["unseen"])] for u in units]))

    out = []
    for key, label in ARMS:
        rec = float(np.mean([u[key] for u in units]))
        # paired comparison against the recommendation, ties shown
        better = sum(1 for u in units if u["shared"] > u[key])
        worse = sum(1 for u in units if u["shared"] < u[key])
        out.append({"key": key, "label": label, "apcer": base - rec,
                    "recovery": rec, "better": better,
                    "equal": len(units) - better - worse, "worse": worse})
    return base, out


def latex(base: float, rs: list[dict]) -> str:
    lines = [f"reduced vocabulary, nothing added & {base:.1f} & --- & --- & --- "
             r"\\", r"\midrule"]
    for r in rs:
        if r["key"] == "shared":
            lines.append(f"{r['label']} & \\textbf{{{r['apcer']:.1f}}} & --- & "
                         r"--- & --- \\")
        else:
            lines.append(f"{r['label']} & {r['apcer']:.1f} & {r['better']} & "
                         f"{r['equal']} & {r['worse']} " + r"\\")
        if r["key"] == "shared":
            lines.insert(len(lines) - 1, r"\midrule")
    return "\n".join(lines)


def selftest() -> None:
    base, rs = rows()
    by = {r["key"]: r for r in rs}

    # the starting point and the scale the main paper states
    assert abs(base - 52.2) < 0.1, base
    assert abs(by["shared"]["recovery"] - 18.4) < 0.05
    assert abs(by["random"]["recovery"] - 10.9) < 0.05
    assert abs(by["most_diff"]["recovery"] - 2.3) < 0.05
    assert abs(by["raw_nc"]["recovery"] - 20.8) < 0.05
    assert abs(by["oracle"]["recovery"] - 21.1) < 0.05

    # rows must be ordered by decreasing APCER: that is what makes the
    # table a scale readable from top to bottom
    ap = [r["apcer"] for r in rs]
    assert ap == sorted(ap, reverse=True), [round(x, 1) for x in ap]

    # every paired comparison sums to 10
    for r in rs:
        assert r["better"] + r["equal"] + r["worse"] == 10, r
    # the oracle cannot be beaten, by definition
    assert by["oracle"]["better"] == 0, by["oracle"]
    # ...and the recommendation equals it on half the unseen PAIs
    assert by["oracle"]["equal"] == 5, by["oracle"]
    # the share of the oracle the main paper quotes
    frac = 100 * by["shared"]["recovery"] / by["oracle"]["recovery"]
    assert abs(frac - 87) < 1, frac
    print(f"selftest ok — start {base:.1f}, X-FPAD {by['shared']['apcer']:.1f}, "
          f"oracle {by['oracle']['apcer']:.1f} ({frac:.0f}% of the maximum)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    base, rs = rows()
    print(f"reduced vocabulary: APCER {base:.2f}\n")
    print(f"{'strategy':<34}{'APCER':>8}{'recovery':>10}"
          f"{'better':>8}{'equal':>7}{'worse':>7}")
    for r in rs:
        print(f"{r['label'][:34]:<34}{r['apcer']:>8.1f}{r['recovery']:>10.1f}"
              f"{r['better']:>8}{r['equal']:>7}{r['worse']:>7}")
    print(f"\nX-FPAD reaches "
          f"{100 * rs[2]['recovery'] / rs[4]['recovery']:.0f}% of the oracle")
    print("\n--- LaTeX body ---")
    print(latex(base, rs))


if __name__ == "__main__":
    main()
