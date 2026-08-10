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
CTX = REPO / "outputs" / "context_stability.json"

# reading order: what you get without training a geometry, then what you get by
# training one other than ours, X-FPAD last
ROWS = [
    ("raw", r"1280-D embedding, nearest centroid", "post-hoc"),
    # each reduction twice: fitted jointly on the training and unseen samples,
    # and fitted on the training set alone and asked to place the unseen ones
    # afterwards. Printing the pair is what keeps the comparison from resting on
    # a protocol the alternatives never chose.
    ("pca", r"PCA $\rightarrow$ 2-D, joint fit", "post-hoc"),
    ("pca_train", r"PCA $\rightarrow$ 2-D, fitted on the training set", "post-hoc"),
    ("tsne", r"t-SNE $\rightarrow$ 2-D, joint fit", "post-hoc"),
    ("tsne_train", r"t-SNE $\rightarrow$ 2-D, fitted on the training set", "post-hoc"),
    ("umap", r"UMAP $\rightarrow$ 2-D, joint fit", "post-hoc"),
    ("umap_train", r"UMAP $\rightarrow$ 2-D, fitted on the training set", "post-hoc"),
    ("xfpad_radialonly", r"radial loss only, no angular term", "trained"),
    ("xfpad_cosface", r"free prototypes, CosFace margin", "trained"),
    ("xfpad_arcface", r"free prototypes, ArcFace margin", "trained"),
    ("xfpad", r"\textbf{X-FPAD}", "trained"),
]


def data() -> dict:
    assert SRC.exists(), f"{SRC} missing; run baselines_predictiveness.py"
    return json.loads(SRC.read_text(encoding="utf-8"))


def context() -> dict:
    assert CTX.exists(), f"{CTX} missing; run context_stability.py"
    return json.loads(CTX.read_text(encoding="utf-8"))


def body() -> str:
    d, ctx = data(), context()
    out, group = [], None
    for key, label, grp in ROWS:
        r = d[key]
        if grp != group:
            head = ("Reductions of the frozen embedding"
                    if grp == "post-hoc" else "Geometric encoders trained on it")
            out.append(r"\addlinespace[2pt]" if group else "")
            out.append(rf"\multicolumn{{7}}{{l}}{{\cellcolor{{gray!10}}\textit{{{head}}}}} \\")
            group = grp
        c = ctx[key]
        n, hits = r["directional_n"], r["directional_hits"]
        ent = f"{r['entropy']:.2f}"
        rho = f"{r['spearman']:+.3f}"
        sign = f"{hits} of {n}" if n else "--"
        sel = f"{100 * r['marginal_inside_floor']:.1f}\\%"
        marg = f"{c['median_margin']:.2f}"
        batch = f"{c['changed']} of {c['total']}"
        if key == "xfpad":
            # \textbf does not embolden mathematics: inside $...$ it takes
            # \mathbf, and the rho cell is math because of the sign
            ent, sign, sel, marg, batch = (
                rf"\textbf{{{x}}}" for x in (ent, sign, sel, marg, batch))
            rho = rf"\mathbf{{{rho}}}"
        out.append(f"{label} & {ent} & ${rho}$ & {sign} & {sel} & {marg} "
                   f"& {batch}" + r" \\")
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
    # three readings name an anchor that then moves against them, and the
    # 1280-D source is not one of them
    wrong = [k for k, _l, _g in ROWS
             if d[k]["directional_hits"] < d[k]["directional_n"]]
    assert wrong == ["pca_train", "umap", "umap_train"], wrong
    for key in ("raw", "pca", "tsne", "tsne_train", "xfpad_cosface",
                "xfpad_arcface"):
        assert d[key]["directional_hits"] == d[key]["directional_n"], key
    # neither protocol lifts a projection to the manifold, and neither is
    # systematically the kinder of the two: the joint fit helps PCA and t-SNE
    # and holds UMAP back. The text says exactly this and nothing more.
    assert d["pca"]["spearman"] > d["pca_train"]["spearman"], d["pca"]
    assert d["tsne"]["spearman"] > d["tsne_train"]["spearman"], d["tsne"]
    assert d["umap"]["spearman"] < d["umap_train"]["spearman"], d["umap"]
    # \textbf does not embolden mathematics, and the rho cell is in math mode
    assert r"\textbf{$" not in body(), "bold applied outside mathematics"
    assert r"$\mathbf{+0.523}$" in body(), "the X-FPAD row is not bold"
    # the two columns added to the table: a reading nobody can move, and one
    # decisive enough to be worth moving. Neither separates X-FPAD on its own
    # --- the free-prototype encoders match both --- so the table has to carry
    # them next to rho rather than instead of it.
    c = context()
    assert all(len(row.split("&")) == 7 for row in body().splitlines()
               if "multicolumn" not in row and "&" in row), "column count"
    # the batch column separates the protocols, not the methods: everything
    # that places a new point through a map it has already settled holds, and
    # only the joint fit moves
    for key in ("xfpad", "xfpad_cosface", "xfpad_arcface", "raw",
                "pca_train", "tsne_train", "umap_train"):
        assert c[key]["changed"] == 0, key
    for key in ("pca", "tsne", "umap"):
        assert c[key]["changed"] > 0, key
    assert c["umap"]["changed"] > c["tsne"]["changed"] > 0, c
    assert c["xfpad_arcface"]["median_margin"] > c["xfpad"]["median_margin"], \
        "ArcFace is the more decisive of the two; the text must not claim otherwise"
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
