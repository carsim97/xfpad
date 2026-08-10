"""The two thresholds of the Phase 3a reading, recomputed over a range.

p_{u,k} >= 0.30 separates the anchors that carry a prediction from the marginal
ones; the noise floor is placed at the first tenth above every matched-size
control. Neither is worth defending on its own, because what matters is how
much either one moves the reading, and that is measurable: the reading is
redone over a range and one looks at what changes.

Only one of the two sensitivities earns a table. The attribution threshold
decides HOW MANY units are read, and the answer is invariant -- at every value
in the range each selected unit shifts in the predicted direction -- so a
sentence carries it. The noise floor decides which shifts count as effects, and
there something does move.

Usage
-----
    python scripts/emit_supp_thresholds.py
    python scripts/emit_supp_thresholds.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts._protocol import NOISE_FLOOR, P_THRESHOLD  # noqa: E402
from scripts.plot_phase3_scatter import clusters, SRC  # noqa: E402

P_GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
FLOOR_GRID = (0.8, 1.0, 1.2, 1.3, 1.5, 2.0, 2.5)
P_PAPER, FLOOR_PAPER = P_THRESHOLD, NOISE_FLOOR


def cl():
    return clusters(json.loads(SRC.read_text(encoding="utf-8"))["records"])


def attribution_rows(units) -> list[dict]:
    out = []
    for th in P_GRID:
        sel = [c for c in units if c["p_uk"] >= th]
        ok = sum(1 for c in sel if c["effect"] > 0)
        out.append({"value": th, "n": len(sel), "ok": ok})
    return out


def floor_rows(units) -> list[dict]:
    pred = [c for c in units if c["p_uk"] >= P_PAPER]
    marg = [c for c in units if c["p_uk"] < P_PAPER]
    out = []
    for th in FLOOR_GRID:
        cleared = sum(1 for c in pred if abs(c["effect"]) >= th)
        # Both columns count the same thing on the two groups -- units above the
        # floor -- so they compare without a change of unit; the denominators
        # appear once, in the header.
        out_marg = sum(1 for c in marg if abs(c["effect"]) >= th)
        inside = 100 * (len(marg) - out_marg) / len(marg)
        out.append({"value": th, "cleared": cleared, "n": len(pred),
                    "marg_cleared": out_marg, "n_marg": len(marg),
                    "inside": inside})
    return out


def latex(units) -> str:
    """Table body: the noise floor only."""
    rows = []
    for r in floor_rows(units):
        mark = r"\rowcolor{gray!12} " if r["value"] == FLOOR_PAPER else ""
        rows.append(f"{mark}${r['value']:.1f}$ & {r['cleared']} & "
                    f"{r['marg_cleared']} " + r"\\")
    return "\n".join(rows)


def attribution_sentence(units) -> str:
    """The numbers the prose quotes in place of the suppressed table block."""
    a = attribution_rows(units)
    seq = ", ".join(str(r["n"]) for r in a)
    same = [r["value"] for r in a if r["n"] == next(
        x["n"] for x in a if x["value"] == P_PAPER)]
    return (f"thresholds {P_GRID} -> units selected {seq}; "
            f"predicted direction holds throughout; "
            f"same set as the paper at {same}")


def selftest() -> None:
    units = cl()
    a, f = attribution_rows(units), floor_rows(units)

    # the predicted direction holds over the WHOLE attribution range
    assert all(r["ok"] == r["n"] for r in a), [(r["value"], r["ok"], r["n"]) for r in a]
    # and the published values come from here
    p30 = next(r for r in a if r["value"] == P_PAPER)
    assert p30["n"] == 11 and p30["ok"] == 11, p30
    # 0.25 and 0.30 select the same set: the threshold is not on a knife edge
    p25 = next(r for r in a if r["value"] == 0.25)
    assert p25["n"] == p30["n"], (p25, p30)
    # the count above the noise is constant from 1.0 to the adopted value; one
    # more joins at 0.8, one drops out at 1.5
    stable = [r for r in f if 1.0 <= r["value"] <= FLOOR_PAPER]
    assert {r["cleared"] for r in stable} == {8}, stable
    fl = next(r for r in f if r["value"] == FLOOR_PAPER)
    assert abs(fl["inside"] - 89.4) < 0.05, fl
    assert [r["n"] for r in a] == [13, 11, 11, 9, 9, 5], [r["n"] for r in a]
    # both columns count units above the floor
    assert all(r["marg_cleared"] + round(r["inside"] * r["n_marg"] / 100)
               == r["n_marg"] for r in f), f
    assert next(r["marg_cleared"] for r in f if r["value"] == FLOOR_PAPER) == 5
    # the control group thins out as the floor rises, the predicted one does not
    marg_seq = [r["marg_cleared"] for r in f]
    assert marg_seq == sorted(marg_seq, reverse=True) and marg_seq[0] > 4 * marg_seq[-1]
    body = latex(units)
    assert "Attribution" not in body and body.count(r"\\") == len(FLOOR_GRID), body
    assert "%" not in body, "the table must not mix counts and percentages"
    print(f"selftest ok — attribution: correct direction at all {len(a)} thresholds, "
          f"{[r['n'] for r in a]} units selected; "
          f"noise: {fl['cleared']}/{fl['n']} constant from 1.0 to the adopted value")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    units = cl()
    if args.selftest:
        selftest()
        return
    print(latex(units))
    print("\n% " + attribution_sentence(units))


if __name__ == "__main__":
    main()
