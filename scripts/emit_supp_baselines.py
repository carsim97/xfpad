"""LaTeX body of the all-materials baselines the shifts are measured from.

Delta/sigma is a ratio: without the level it starts from, +2 could be a detector
going from 5% to 15% or one going from 80% to 88%. The ablation table of the
main paper carries a 'base' column condensed over the audited systems; this
opens it up, with the standard deviation that enters the denominator of
Delta/sigma, so that every value of the complete sweep can be read against the
level it moved from.

Usage
-----
    python scripts/emit_supp_baselines.py
    python scripts/emit_supp_baselines.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LOMO = REPO / "outputs" / "lomo"
sys.path.insert(0, str(REPO))
from scripts._protocol import AUDITED, short  # noqa: E402


def baselines() -> dict:
    """{(scanner, PAI): {system: (mean, std)}}"""
    out: dict = {}
    for scanner in ("greenbit", "dermalog"):
        for bb in AUDITED:
            p = LOMO / f"phase3_{scanner}_{bb}_baseline.json"
            assert p.exists(), p
            for u, v in json.loads(p.read_text(encoding="utf-8"))[
                    "aggregated"]["per_pai"].items():
                out.setdefault((scanner, u), {})[bb] = (float(v["mean"]),
                                                        float(v["std"]))
    return out


def latex(acc: dict) -> str:
    """Green Bit before Dermalog, as everywhere else in the document."""
    order = {"greenbit": 0, "dermalog": 1}
    rows, sensor = [], None
    for key in sorted(acc, key=lambda k: (order[k[0]], k[1])):
        if key[0] != sensor:
            sensor = key[0]
            name = "Green Bit" if sensor == "greenbit" else "Dermalog"
            if rows:
                rows.append(r"\addlinespace[2pt]")
            rows.append(r"\multicolumn{4}{l}{\cellcolor{gray!10}\textbf{" + name
                        + r"}} \\")
        cells = " & ".join(f"${m:.2f} \\pm {s:.2f}$"
                           for m, s in (acc[key][b] for b in AUDITED))
        rows.append(f"{short(key[1])} & {cells} " + r"\\")
    return "\n".join(rows)


def selftest() -> None:
    acc = baselines()
    assert len(acc) == 10, len(acc)
    assert all(len(v) == 4 for v in acc.values())
    # the 'base' column of Table IV is the average over the audited systems
    for key, expect in ((("greenbit", "Mix1 Consensual"), 5.5),
                        (("greenbit", "Body Double ScreenSpoof"), 23.0),
                        (("dermalog", "GLS20 ScreenSpoof"), 80.6),
                        (("dermalog", "RFast30 Consensual"), 5.1)):
        got = float(np.mean([acc[key][b][0] for b in AUDITED]))
        assert abs(got - expect) < 0.05, f"{key}: {got:.2f} instead of {expect}"
    # the saturated regime the main paper invokes is there in the data
    dm_ss = [acc[k] for k in acc if k[0] == "dermalog" and "ScreenSpoof" in k[1]]
    assert all(m > 70 for v in dm_ss for m, _s in v.values()), "ScreenSpoof not saturated"
    # inside the single float the panels must list the sensors in one order
    body = latex(acc)
    assert body.index("Green Bit") < body.index("Dermalog"), \
        "the sensor order is not the one used in the rest of the document"
    print(f"selftest ok — {len(acc)} unseen PAIs, averages consistent with "
          f"the 'base' column of the main paper")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    acc = baselines()
    if args.selftest:
        selftest()
        return
    print(latex(acc))


if __name__ == "__main__":
    main()
