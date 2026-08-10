"""LaTeX body of the Phase 3a directional-ablation table.

One row per analysis unit -- sensor, removed material, unseen PAI -- carrying
the attribution weight the removal was predicted from, the all-materials APCER
it starts from, and the effect size averaged over the audited systems. The
per-system values live in the supplementary sweep.

Each row carries its baseline because Delta/sigma alone is a ratio: a unit
operating near the ceiling compresses Delta, and without the absolute level
beside it the effect size is not readable.

Usage
-----
    python scripts/emit_phase3_table.py
    python scripts/emit_phase3_table.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CORR = REPO / "outputs" / "point_b_correlation.json"
LOMO = REPO / "outputs" / "lomo"

# The four audited systems enter on equal terms: trained by us on the same
# task, under the same protocol and the same ablations. The analysis unit is
# their average.
sys.path.insert(0, str(REPO))
from scripts._protocol import AUDITED, NOISE_FLOOR, P_THRESHOLD, short  # noqa: E402


def baselines(scanner: str) -> dict[str, float]:
    """Baseline APCER per PAI, averaged over the audited systems."""
    acc: dict[str, list[float]] = defaultdict(list)
    for bb in AUDITED:
        p = LOMO / f"phase3_{scanner}_{bb}_baseline.json"
        assert p.exists(), f"{p} missing"
        per_pai = json.loads(p.read_text(encoding="utf-8"))["aggregated"]["per_pai"]
        for pai, node in per_pai.items():
            v = node["APCER"]["mean"] if isinstance(node.get("APCER"), dict) else node["mean"]
            acc[pai].append(float(v))
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    assert all(len(v) == len(AUDITED) for v in acc.values()), \
        f"{scanner}: a PAI has no baseline on every audited system"
    return out


def clusters() -> list[dict]:
    recs = json.loads(CORR.read_text(encoding="utf-8"))["records"]
    acc = defaultdict(lambda: {"ds": [], "d": [], "p": None})
    for r in recs:
        k = (r["scanner"], r["removed_material"], r["unseen"])
        acc[k]["ds"].append(r["delta_over_sigma"])
        acc[k]["d"].append(r["delta"])
        acc[k]["p"] = float(r["p_uk"])
    out = []
    for (sc, rem, uns), v in acc.items():
        assert len(v["ds"]) == len(AUDITED), f"{sc}/{rem}/{uns}: an audited system is missing"
        out.append({"scanner": sc, "removed": rem, "unseen": uns, "p": v["p"],
                    "delta": float(np.mean(v["d"])), "ds": float(np.mean(v["ds"]))})
    assert len(out) == 58, f"expected 58 units, found {len(out)}"
    return out


def emit() -> str:
    cl = clusters()
    direc = [c for c in cl if c["p"] >= P_THRESHOLD]
    assert len(direc) == 11, f"expected 11 predicted units, found {len(direc)}"
    assert all(c["ds"] > 0 for c in direc), "a prediction has the wrong sign"

    parts = []
    for i, sc in enumerate(("greenbit", "dermalog")):
        base = baselines(sc)
        rows = sorted([c for c in direc if c["scanner"] == sc], key=lambda c: -c["p"])
        label = "Green Bit" if sc == "greenbit" else "Dermalog"
        head = (("\\addlinespace[2pt]\n" if i else "")
                + f"\\multicolumn{{5}}{{l}}{{\\cellcolor{{gray!10}}\\textbf{{{label}}}}} \\\\\n")
        body = []
        for c in rows:
            shade = "\\cellcolor{green!18}" if c["ds"] >= NOISE_FLOOR else ""
            body.append(f"{short(c['removed']):<12}& {short(c['unseen']):<14}& {c['p']:.2f} "
                        f"& {base[c['unseen']]:.1f} & {shade}${c['ds']:+.2f}$ \\\\")
        parts.append(head + "\n".join(body))
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    body = emit()
    if args.selftest:
        print(f"  {body.count(chr(92) + chr(92))} rows generated, selftest ok")
        return
    print(body)


if __name__ == "__main__":
    main()
