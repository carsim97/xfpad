"""LaTeX bodies of the Phase 2 attribution tables, from the analysis JSONs.

The tables are emitted from the JSONs the analysis produces and never retyped
into the LaTeX source: that is the only way the weights printed in a table and
the weights Phase 3a pairs with the observed effect cannot drift apart.

Two bodies:
  * main  -- top four anchors, aggregated tail, no standard deviations
  * suppl -- the same anchors with mean +/- standard deviation

Usage
-----
    python scripts/emit_phase2_tables.py            # print both bodies
    python scripts/emit_phase2_tables.py --selftest # checks only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SCANNERS = ("greenbit", "dermalog")
N_TOP = 4
# The tail is the sum of the anchors NOT listed, so the printed row sums to one
# like the attribution vector itself: a threshold on the weight would drop the
# anchors just above it that do not make the top four.

ABBR = {"Wood Glue": "W.G.", "Latex": "Lx", "Body Double": "B.D.",
        "RProFast": "R.F.", "Ecoflex": "Ecx", "RPro10": "RPr", "Latex V2": "LxV2"}
PAI_ABBR = {"Consensual": "Cons.", "ScreenSpoof": "S.S."}


def _short_pai(name: str) -> str:
    for long, short in PAI_ABBR.items():
        name = name.replace(long, short)
    for long, short in (("Body Double", "B.D."), ("Elmer's Glue", "E.G.")):
        name = name.replace(long, short)
    return name


def load(scanner: str) -> dict:
    p = REPO / "outputs" / f"phase2_{scanner}_intra.json"
    assert p.exists(), f"{p} missing"
    return json.loads(p.read_text(encoding="utf-8"))["aggregated"]["attribution"]


def rows(scanner: str) -> tuple[list[str], list[str]]:
    attr = load(scanner)
    main, supp = [], []
    for pai, node in attr.items():
        mean, std = node["mean"], node["std"]
        top = sorted(mean.items(), key=lambda kv: -kv[1])[:N_TOP]
        tail = sum(v for k, v in mean.items() if k not in {m for m, _ in top})
        assert abs(sum(mean.values()) - 1.0) < 1e-6, f"{pai}: weights do not sum to 1"
        # what reaches the page must sum to one like the true vector, up to the
        # rounding of the five printed cells
        shown = sum(round(v, 2) for _m, v in top) + round(tail, 2)
        assert abs(shown - 1.0) < 0.02, f"{pai}: the printed row sums to {shown:.2f}"

        cells = " & ".join(f"{ABBR[m]} {v:.2f}" for m, v in top)
        tail_s = f"{tail:.2f}" if tail >= 0.005 else "--"
        main.append(f"{_short_pai(pai):<18}& {cells} & {tail_s} \\\\")

        cells_sd = " & ".join(f"{ABBR[m]} ${v:.2f} \\pm {std[m]:.2f}$" for m, v in top)
        supp.append(f"{_short_pai(pai):<18}& {cells_sd} & {tail_s} \\\\")
    return main, supp


def emit() -> tuple[str, str]:
    main_parts, supp_parts = [], []
    for i, sc in enumerate(SCANNERS):
        head = (("\\midrule\n" if i else "")
                + f"\\multicolumn{{6}}{{l}}{{\\textit{{{sc.replace('greenbit','Green Bit').replace('dermalog','Dermalog')}}}}} \\\\\n"
                + "\\cmidrule(lr){1-6}\n")
        m, s = rows(sc)
        main_parts.append(head + "\n".join(m))
        supp_parts.append(head + "\n".join(s))
    return "\n".join(main_parts), "\n".join(supp_parts)


def checks() -> None:
    """The statements in the text that depend on these numbers."""
    gb, dm = load("greenbit"), load("dermalog")
    marg = max(max(gb[p]["mean"][m] for p in gb) for m in ("Latex V2", "RPro10"))
    ss = max(dm[p]["mean"][m] for p in dm if "ScreenSpoof" in p
             for m in ("RProFast", "Latex V2", "RPro10"))
    diffuse = max(max(gb[p]["mean"].values())
                  for p in ("Body Double Consensual", "Elmer's Glue Consensual",
                            "Mix1 ScreenSpoof"))
    print(f"  Latex V2 / RPro10, max on Green Bit : {marg:.3f}  (text: 'below 0.08')")
    print(f"  Dermalog ScreenSpoof, remainder     : {ss:.3f}  (text: 'below 0.13')")
    print(f"  diffuse group, highest anchor       : {diffuse:.3f}  (text: 'no material reaches 0.45')")
    assert marg < 0.08 and ss < 0.13 and diffuse < 0.45, \
        "a statement in the text no longer holds: reword it, do not round it"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    checks()
    if args.selftest:
        emit()
        print("  selftest ok")
        return
    m, s = emit()
    print("\n% ---- Table III (main) ----\n" + m)
    print("\n% ---- Table S2 (supplementary) ----\n" + s)


if __name__ == "__main__":
    main()
