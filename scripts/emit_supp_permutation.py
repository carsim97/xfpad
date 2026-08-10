"""Stability of the attribution under permutation of the prototypes.

Materials are assigned to angular sectors by label index, so the attribution
could in principle follow that ordering rather than the geometry. The assignment
is permuted, the encoder retrained, and the anchors each unseen PAI receives are
compared with the identity layout.

Usage
-----
    python scripts/emit_supp_permutation.py
    python scripts/emit_supp_permutation.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO))
from scripts._protocol import short  # noqa: E402


def per_unseen(scanner: str) -> dict:
    """{PAI: (permutations preserving top-1, preserving top-2, n)}"""
    d = json.loads((REPO / "outputs" / "point_d" /
                    f"permutation_{scanner}.json").read_text(encoding="utf-8"))
    acc: dict[str, list[int]] = {}
    for perm, units in d["per_permutation_vs_identity"].items():
        for u, v in units.items():
            a = acc.setdefault(u, [0, 0, 0])
            a[0] += int(v["top1_match"])
            a[1] += int(v["top2_match"])
            a[2] += 1
    return acc, d["summary"]


def latex() -> str:
    rows = []
    for scanner, name in (("greenbit", "Green Bit"), ("dermalog", "Dermalog")):
        acc, _ = per_unseen(scanner)
        if rows:
            rows.append(r"\addlinespace[2pt]")
        rows.append(r"\multicolumn{3}{l}{\cellcolor{gray!10}\textbf{" + name
                    + r"}} \\")
        for u in sorted(acc):
            t1, t2, n = acc[u]
            rows.append(f"{short(u)} & {t1}/{n} & {t2}/{n} " + r"\\")
    return "\n".join(rows)


def selftest() -> None:
    for scanner in ("greenbit", "dermalog"):
        acc, summary = per_unseen(scanner)
        assert summary["top1_match_rate"] == 1.0, summary
        # the dominant anchor survives EVERY permutation, on every unseen PAI
        for u, (t1, _t2, n) in acc.items():
            assert t1 == n, f"{scanner}/{u}: top-1 {t1}/{n}"
        assert summary["n_permutations"] == 9, summary
    # the secondary structure holds only in part, which is why
    # the main paper confines its reading to the dominant anchor
    gb, _ = per_unseen("greenbit")
    dm, _ = per_unseen("dermalog")
    frac_gb = sum(t2 for _t1, t2, _n in gb.values()) / sum(n for *_x, n in gb.values())
    frac_dm = sum(t2 for _t1, t2, _n in dm.values()) / sum(n for *_x, n in dm.values())
    assert frac_gb < frac_dm < 1.0, (frac_gb, frac_dm)
    print(f"selftest ok — top-1 preserved under every permutation; "
          f"top-2 {100*frac_gb:.0f}% Green Bit, {100*frac_dm:.0f}% Dermalog")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    print(latex())


if __name__ == "__main__":
    main()
