"""LaTeX body of the complete leave-one-material-out sweep.

The main paper tabulates the 11 units on which Phase 2 issues a prediction; this
generates the supplementary table that carries all 58, each with the
Delta/sigma of the four audited systems and the average that serves as the unit
of analysis.

One row per (removed material, unseen PAI), grouped by removed material, so that
a whole ablation reads as a block -- which is the reading the specificity
control needs: when the removed material is not implicated, its entire block
stays flat.

Usage
-----
    python scripts/emit_supp_sweep.py
    python scripts/emit_supp_sweep.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "outputs" / "point_b_correlation.json"
sys.path.insert(0, str(REPO))
from scripts._protocol import AUDITED, P_THRESHOLD, short  # noqa: E402

# presentation order: the materials Phase 2 implicates come first
ORDER = {"greenbit": ["Wood Glue", "Body Double", "Latex", "RProFast",
                      "Ecoflex", "Latex V2", "RPro10"],
         "dermalog": ["RProFast", "Latex", "Latex V2", "RPro10"]}


def units() -> dict:
    """{(scanner, removed, unseen): {system: d/s, 'mean': ..., 'p': ...}}"""
    acc: dict[tuple, dict] = defaultdict(dict)
    for r in json.loads(SRC.read_text(encoding="utf-8"))["records"]:
        key = (r["scanner"], r["removed_material"], r["unseen"])
        acc[key][r["backbone"]] = float(r["delta_over_sigma"])
        acc[key]["p"] = float(r["p_uk"])
    for k, v in acc.items():
        assert all(b in v for b in AUDITED), f"{k}: missing systems"
        v["mean"] = float(np.mean([v[b] for b in AUDITED]))
    return acc


def latex(scanner: str, acc: dict) -> str:
    rows, first = [], True
    for material in ORDER[scanner]:
        block = sorted((k for k in acc if k[0] == scanner and k[1] == material),
                       key=lambda k: k[2])
        if not block:
            continue
        if not first:
            rows.append(r"\addlinespace[2pt]")
        first = False
        rows.append(r"\multicolumn{6}{l}{\cellcolor{gray!10}\textit{Without "
                    + material + r"}} \\")
        for key in block:
            v = acc[key]
            cells = " & ".join(f"${v[b]:+.2f}$" for b in AUDITED)
            # \textbf around mathematics does not embolden it: the bold has to
            # go inside math mode, or the caption promises a mark the table
            # does not show
            mean = f"$\\mathbf{{{v['mean']:+.2f}}}$" if v["p"] >= P_THRESHOLD \
                else f"${v['mean']:+.2f}$"
            rows.append(f"{short(key[2])} & {v['p']:.2f} & {cells} & {mean} " + r"\\")
    return "\n".join(rows)


def selftest() -> None:
    acc = units()
    assert len(acc) == 58, len(acc)
    gb = [k for k in acc if k[0] == "greenbit"]
    dm = [k for k in acc if k[0] == "dermalog"]
    assert len(gb) == 42 and len(dm) == 16, (len(gb), len(dm))
    pred = [k for k in acc if acc[k]["p"] >= P_THRESHOLD]
    assert len(pred) == 11, len(pred)
    # the averages must match the values the main paper publishes in Table IV
    for key, expect in ((("greenbit", "Wood Glue", "Mix1 Consensual"), 16.02),
                        (("dermalog", "RProFast", "RFast30 Consensual"), 3.73),
                        (("dermalog", "Latex", "GLS20 ScreenSpoof"), 3.00)):
        got = acc[key]["mean"]
        assert abs(got - expect) < 0.005, f"{key}: {got:.2f} instead of {expect}"
    # every material in the presentation order exists, and the order is the
    # highest attribution weight the material receives, as the prose states
    for scanner, mats in ORDER.items():
        have = {k[1] for k in acc if k[0] == scanner}
        assert set(mats) == have, (scanner, set(mats) ^ have)
        top = [max(acc[k]["p"] for k in acc if k[0] == scanner and k[1] == m)
               for m in mats]
        assert top == sorted(top, reverse=True), (scanner, list(zip(mats, top)))
        # the consequence visible in the table, which the prose states: blocks
        # holding at least one predicted unit all come before those holding none
        has = [t >= P_THRESHOLD for t in top]
        assert has == sorted(has, reverse=True), (scanner, list(zip(mats, has)))
    # the mark on predicted units has to be genuinely bold
    body = "\n".join(latex(sc, acc) for sc in ("greenbit", "dermalog"))
    assert r"\textbf{$" not in body, "bold applied outside math mode"
    assert body.count(r"$\mathbf{") == len(pred), \
        f"{body.count(chr(36) + chr(92) + 'mathbf{')} bold cells instead of {len(pred)}"
    print(f"selftest ok — {len(acc)} units ({len(gb)} Green Bit, {len(dm)} "
          f"Dermalog), {len(pred)} predicted and marked")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    acc = units()
    if args.selftest:
        selftest()
        return
    for scanner in ("greenbit", "dermalog"):
        print(f"%--- {scanner} ---")
        print(latex(scanner, acc))
        print()


if __name__ == "__main__":
    main()
