"""Body of Table S7: the Phase 3a reading applied to other representations.

The diagnostic baselines -- nearest centroid in the 1280-D space, post-hoc
reductions, a radial-only encoder, free prototypes in the CosFace/ArcFace style
-- are not judged by eye. Each is fed the criterion the paper reads Phase 3a
with: how many primary anchors it names, whether the shift carries the sign it
assigns, and whether what it calls marginal stays inside the noise floor.

The X-FPAD row must reproduce the numbers of Section V-C exactly; that is the
check tying this table to the rest of the paper, and it is asserted.

Usage
-----
    python scripts/emit_supp_representations.py
    python scripts/emit_supp_representations.py --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "outputs" / "point_a_predictiveness.json"

# reading order: what you get without training a geometry, then what you get by
# training one other than ours, X-FPAD last
ROWS = [
    ("raw", r"1280-D embedding, nearest centroid", "post-hoc"),
    ("pca", r"PCA $\rightarrow$ 2-D", "post-hoc"),
    ("tsne", r"t-SNE $\rightarrow$ 2-D", "post-hoc"),
    ("umap", r"UMAP $\rightarrow$ 2-D", "post-hoc"),
    ("xfpad_radialonly", r"radial loss only, no angular term", "trained"),
    ("xfpad_cosface", r"free prototypes, CosFace margin", "trained"),
    ("xfpad_arcface", r"free prototypes, ArcFace margin", "trained"),
    ("xfpad", r"\textbf{X-FPAD}", "trained"),
]


def data() -> dict:
    assert SRC.exists(), f"{SRC} missing; run baselines_predictiveness.py"
    return json.loads(SRC.read_text(encoding="utf-8"))


def body() -> str:
    d = data()
    out, group = [], None
    for key, label, grp in ROWS:
        r = d[key]
        if grp != group:
            head = ("Reductions of the frozen embedding"
                    if grp == "post-hoc" else "Geometric encoders trained on it")
            out.append(r"\addlinespace[2pt]" if group else "")
            out.append(rf"\multicolumn{{5}}{{l}}{{\cellcolor{{gray!10}}\textit{{{head}}}}} \\")
            group = grp
        n, hits = r["directional_n"], r["directional_hits"]
        ent = f"{r['entropy']:.2f}"
        rho = f"{r['spearman']:+.3f}"
        sign = f"{hits} of {n}" if n else "--"
        sel = f"{100 * r['marginal_inside_floor']:.1f}\\%"
        if key == "xfpad":
            # \textbf does not embolden mathematics: inside $...$ it takes
            # \mathbf, and the rho cell is math because of the sign
            ent, sign, sel = (rf"\textbf{{{c}}}" for c in (ent, sign, sel))
            rho = rf"\mathbf{{{rho}}}"
        out.append(f"{label} & {ent} & ${rho}$ & {sign} & {sel}" + r" \\")
    return "\n".join(x for x in out if x)


def selftest() -> None:
    d = data()
    x = d["xfpad"]
    # the X-FPAD row is the reading published in Section V-C
    assert x["directional_n"] == 11 and x["directional_hits"] == 11, x
    assert abs(x["spearman"] - 0.523) < 0.001, x["spearman"]
    assert round(100 * x["marginal_inside_floor"], 1) == 89.4, x
    assert x["n_clusters"] == 58, x
    # every alternative is less predictive, which is the only claim the section
    # makes: were one to beat it, the text would need rewriting
    for key, _lbl, _g in ROWS:
        if key == "xfpad":
            continue
        assert d[key]["spearman"] < x["spearman"], (key, d[key]["spearman"])
        assert d[key]["directional_n"] <= x["directional_n"], key
    # radial-only names no anchor at all and its attribution is flat
    ro = d["xfpad_radialonly"]
    assert ro["directional_n"] == 0 and round(ro["entropy"], 2) == 1.00, ro
    # UMAP is the only one that gets a sign wrong among those naming anchors
    assert d["umap"]["directional_hits"] < d["umap"]["directional_n"], d["umap"]
    for key in ("raw", "pca", "tsne", "xfpad_cosface", "xfpad_arcface"):
        assert d[key]["directional_hits"] == d[key]["directional_n"], key
    # \textbf does not embolden mathematics, and the rho cell is in math mode
    assert r"\textbf{$" not in body(), "bold applied outside mathematics"
    assert r"$\mathbf{+0.523}$" in body(), "the X-FPAD row is not bold"
    print(f"  selftest ok — {len(ROWS)} representations; X-FPAD "
          f"rho {x['spearman']:+.3f}, {x['directional_hits']}/{x['directional_n']} "
          f"units, {100 * x['marginal_inside_floor']:.1f}% of controls inside")


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
