"""Body of Table S8: what the attribution does when the sensor changes.

Each unseen PAI is projected twice -- onto the manifold of its own sensor, and
onto the manifold of the other one -- and read on the two quantities the
directional claim rests on: how much of the cluster stays inside the bona fide
core, where there is no radius to carry a direction, and how decisive the
attribution is once a direction exists.

The two failure modes the main paper distinguishes are visible in these
columns. A cluster that collapses into the core loses the radius; a cluster
that keeps its radius can still lose the direction, its dominant weight falling
towards the uniform value and its entropy rising.

Usage
-----
    python scripts/emit_supp_crosssensor.py
    python scripts/emit_supp_crosssensor.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"

sys.path.insert(0, str(REPO))
from scripts._protocol import P_THRESHOLD, short  # noqa: E402

# For a PAI of sensor X: "own" is the manifold of X, "other" the manifold
# trained on the companion sensor and asked to read X's attacks.
SRC = {
    "dermalog": ("phase2_dermalog_intra.json", "phase2_greenbit_to_dermalog.json"),
    "greenbit": ("phase2_greenbit_intra.json", "phase2_dermalog_to_greenbit.json"),
}

ORDER = {
    "dermalog": ["GLS20 Consensual", "RFast30 Consensual",
                 "GLS20 ScreenSpoof", "RFast30 ScreenSpoof"],
    "greenbit": ["Mix1 Consensual", "Body Double Consensual",
                 "Elmer's Glue Consensual", "Mix1 ScreenSpoof",
                 "Body Double ScreenSpoof", "Elmer's Glue ScreenSpoof"],
}


def entropy(weights: dict) -> float:
    """Attribution entropy, normalised by the vocabulary it is read against."""
    p = [v for v in weights.values() if v > 0]
    assert len(p) > 1, "a single prototype carries no entropy"
    assert abs(sum(weights.values()) - 1.0) < 1e-6, sum(weights.values())
    return -sum(x * math.log(x) for x in p) / math.log(len(p))


def read(name: str) -> dict:
    p = OUT / name
    assert p.exists(), f"{p} missing; run phase2_analysis.py for both directions"
    return json.loads(p.read_text(encoding="utf-8"))["aggregated"]


def cell(agg: dict, pai: str) -> tuple:
    """(share inside the core, dominant weight, dominant anchor, entropy)."""
    z = agg["zones_per_pai"][pai]
    w = agg["attribution"][pai]["mean"]
    anchor, top = max(w.items(), key=lambda kv: kv[1])
    return 100 * z["below_rho_bf"]["mean"], top, anchor, entropy(w)


def rows() -> list:
    out = []
    for scanner, (own_f, other_f) in SRC.items():
        own, other = read(own_f), read(other_f)
        for pai in ORDER[scanner]:
            out.append((scanner, pai, cell(own, pai), cell(other, pai)))
    return out


def body() -> str:
    lines, panel = [], None
    for scanner, pai, o, x in rows():
        if scanner != panel:
            head = "Dermalog PAIs" if scanner == "dermalog" else "Green Bit PAIs"
            lines.append(r"\addlinespace[2pt]" if panel else "")
            lines.append(rf"\multicolumn{{7}}{{l}}{{\cellcolor{{gray!10}}"
                         rf"\textbf{{{head}}}}} \\")
            panel = scanner
        lines.append(f"{short(pai)} & {o[0]:.1f} & {o[1]:.2f} & {o[3]:.2f}"
                     f" & {x[0]:.1f} & {x[1]:.2f} & {x[3]:.2f}" + r" \\")
    return "\n".join(x for x in lines if x)


def selftest() -> None:
    r = rows()
    assert len(r) == 10, len(r)

    own_top = [x[2][1] for x in r]
    other_top = [x[3][1] for x in r]

    # The two vocabularies differ in size, so a dominant weight of 0.44 read
    # against four prototypes is not the 0.43 read against seven. The entropy
    # is normalised by the vocabulary and is the comparable quantity: it rises
    # for every unseen PAI, and so does the share the core absorbs.
    for _s, pai, o, x in r:
        assert x[3] > o[3], (pai, "entropy", o[3], x[3])
        assert x[0] > o[0], (pai, "core", o[0], x[0])

    # Intra-sensor every cluster names an anchor above the threshold Phase 3a
    # reads at. Across sensors most of them still do: the attribution does not
    # announce its own failure, which is the reason the protocol is per-sensor.
    assert min(own_top) >= P_THRESHOLD, min(own_top)
    assert sum(t >= P_THRESHOLD for t in other_top) >= 8, other_top

    by = {p: (o, x) for _s, p, o, x in r}

    # Mechanism 1, collapse: the Dermalog ScreenSpoof clusters lose the radius
    # entirely when read on the Green Bit manifold.
    for pai in ("GLS20 ScreenSpoof", "RFast30 ScreenSpoof"):
        assert by[pai][1][0] > 97.0, (pai, by[pai][1][0])

    # Mechanism 2, flattening: the Consensual variants keep their radius and
    # still lose the anchor -- RProFast intra, RPro10 across, and RPro10 is the
    # material whose removal moves nothing anywhere in the sweep.
    for pai in ("GLS20 Consensual", "RFast30 Consensual"):
        own, other = by[pai]
        assert other[0] < 25.0, (pai, other[0])
        assert own[2] == "RProFast" and other[2] == "RPro10", (pai, own[2], other[2])

    print(f"  selftest ok — {len(r)} unseen PAIs; the entropy rises and the "
          f"core grows on all of them, and "
          f"{sum(t >= P_THRESHOLD for t in other_top)} "
          f"still name an anchor above the threshold across sensors")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    print(body())


if __name__ == "__main__":
    main()
