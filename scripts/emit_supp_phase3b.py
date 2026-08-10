"""LaTeX body of the per-unit Phase 3b table.

The main paper reports the averages and the counts; here the data go down to the
unit: the material recommended, how many of the ten runs choose it, and the
recovery that recommendation obtains against the alternatives.

The paired tests do not make a table of their own. Of the four columns they had,
two -- mean difference and win count -- follow by subtraction from the table
above, and a third is the sign test, which is a function of the same count. What
remains is the Wilcoxon, and only one of its four values says anything the sign
test does not: the comparison against the random choice. One number does not
earn a table, so it lives in the prose and is emitted here for checking.

Usage
-----
    python scripts/emit_supp_phase3b.py
    python scripts/emit_supp_phase3b.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy import stats

REPO = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO))
from scripts._protocol import short  # noqa: E402
SRC = REPO / "outputs" / "point_e_shared_anchor.json"

ARMS = [("random", "random"), ("most_diff", "most dissimilar"),
        ("raw_nc", "nearest centroid, 1280-D"), ("oracle", "oracle")]


def data() -> list[dict]:
    return json.loads(SRC.read_text(encoding="utf-8"))["units"]


def paired(units) -> list[dict]:
    out = []
    for key, label in ARMS:
        diff = [u["shared"] - u[key] for u in units]
        nz = [d for d in diff if d != 0]
        wins = sum(1 for d in nz if d > 0)
        out.append({"key": key, "label": label, "n": len(nz), "wins": wins,
                    "delta": sum(diff) / len(diff),
                    "sign": stats.binomtest(wins, len(nz), 0.5).pvalue,
                    "wilcoxon": stats.wilcoxon(nz).pvalue})
    return out


def latex_units(units) -> str:
    rows, sensor = [], None
    for u in sorted(units, key=lambda u: (u["scanner"], u["unseen"])):
        if u["scanner"] != sensor:
            sensor = u["scanner"]
            name = "Green Bit" if sensor == "greenbit" else "Dermalog"
            if rows:
                rows.append(r"\addlinespace[2pt]")
            rows.append(r"\multicolumn{7}{l}{\cellcolor{gray!10}\textbf{"
                        + name + r"}} \\")
        rows.append(
            f"{short(u['unseen'])} & {u['pick']} & {u['votes']}/{u['seeds']} & "
            f"${u['shared']:.1f}$ & ${u['random']:.1f}$ & "
            f"${u['most_diff']:.1f}$ & ${u['oracle']:.1f}$ " + r"\\")
    return "\n".join(rows)


def latex_tests(units) -> str:
    rows = []
    for r in paired(units):
        rows.append(f"{r['label']} & ${r['delta']:+.1f}$ & {r['wins']} of "
                    f"{r['n']} & ${r['sign']:.3f}$ & ${r['wilcoxon']:.3f}$ " + r"\\")
    return "\n".join(rows)


def selftest() -> None:
    units = data()
    assert len(units) == 10, len(units)
    by = {r["key"]: r for r in paired(units)}
    # the counts the main paper publishes in Table V
    assert (by["random"]["wins"], by["random"]["n"]) == (8, 10), by["random"]
    assert (by["most_diff"]["wins"], by["most_diff"]["n"]) == (9, 9), by["most_diff"]
    assert (by["raw_nc"]["wins"], by["raw_nc"]["n"]) == (0, 2), by["raw_nc"]
    # e i margini
    assert abs(by["random"]["delta"] - 7.5) < 0.05, by["random"]
    assert abs(by["most_diff"]["delta"] - 16.1) < 0.05, by["most_diff"]
    # the Wilcoxon says something the sign test does not: on the comparison
    # against the random choice the two diverge, which is why the main paper
    # does not claim it
    assert by["random"]["sign"] > 0.10 and by["random"]["wilcoxon"] < 0.05, by["random"]
    # The vote separates the outcomes, but it has to be stated so the reader
    # can check it in the table: 'the two furthest from the oracle' is true as
    # a ratio and false as a difference. What the columns show is that the two
    # least-voted units are the only ones an alternative recovers more on.
    lost = {u["unseen"] for u in units
            if max(u["random"], u["most_diff"]) > u["shared"]}
    assert {u["votes"] for u in units if u["unseen"] in lost} == {5, 6}, lost
    assert len(lost) == 2, lost
    strong = [u for u in units if u["votes"] >= 9]
    assert len(strong) == 7, len(strong)
    took = [u for u in strong if abs(u["oracle"] - u["shared"]) < 0.05]
    assert len(took) == 5, len(took)
    assert max(u["oracle"] - u["shared"] for u in strong) <= 1.3 + 1e-9
    # the share the main paper publishes for units with at least eight votes
    ratios = [u["shared"] / u["oracle"] for u in units
              if u["votes"] >= 8 and u["oracle"] > 0.05]
    assert round(100 * min(ratios)) == 94, min(ratios)
    eg = next(u for u in units if u["unseen"] == "Elmer's Glue Consensual")
    assert (round(eg["shared"], 1), round(eg["oracle"], 1)) == (6.3, 29.5), eg
    print(f"selftest ok — {len(units)} unseen PAIs; on random, sign "
          f"{by['random']['sign']:.3f} contro Wilcoxon "
          f"{by['random']['wilcoxon']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    units = data()
    if args.selftest:
        selftest()
        return
    print("%--- per unit ---")
    print(latex_units(units))
    print("\n%--- paired tests (for checking, no longer a table) ---")
    print(latex_tests(units))


if __name__ == "__main__":
    main()
