"""Diagnostic baselines for the directional attribution.

Compares the X-FPAD manifold attribution p_{u,k} against alternative
feature-space diagnostics computed on the SAME cached backbone embeddings:

  * raw   : nearest-centroid attribution in the original 1280-D f_phi space
            (no dimensionality reduction);
  * pca   : PCA -> 2-D, then the same attribution;
  * tsne  : t-SNE -> 2-D (seeded), then the same attribution;
  * umap  : UMAP -> 2-D (seeded), then the same attribution;
  * xfpad : projection through the trained geometric encoder g_psi (the paper's
            manifold).

All representations feed the identical `analyze_unseen_pais` estimator
(temperature-scaled cosine softmax over empirical centroid directions), so the
comparison isolates the effect of the *representation* on the attribution.

The script is CPU-only by design (set CUDA_VISIBLE_DEVICES="" or rely on the
--device cpu default) so it does not contend with GPU training jobs: it reads
cached .npy features and runs a tiny MLP forward pass for the xfpad case.

Usage
-----
    CUDA_VISIBLE_DEVICES="" python scripts/baselines_attribution.py \
        -c configs/greenbit.yaml \
        --reductions raw pca tsne umap xfpad \
        --save-json outputs/baselines_greenbit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import (  # noqa: E402
    base_parser,
    bona_fide_label,
    known_names,
    known_pairs,
    load_with_overrides,
    unseen_names,
    unseen_pairs,
)
from xfpad.config import load_config  # noqa: E402
from xfpad.data import build_labels  # noqa: E402
from xfpad.metrics import analyze_unseen_pais  # noqa: E402
from xfpad.utils import (  # noqa: E402
    features_path,
    geometric_ckpt,
    get_logger,
    read_split,
    split_path,
)

LOG = get_logger("baselines")


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

def _reduce(name: str,
            feats_train: np.ndarray,
            feats_test: np.ndarray,
            seed: int,
            geo_ckpt: Path | None,
            dropout: float):
    """Return (z_train, z_test) for the requested representation."""
    name = name.lower()
    if name == "raw":
        return feats_train, feats_test

    if name in {"pca", "tsne", "umap"}:
        joint = np.concatenate([feats_train, feats_test], axis=0)
        split = feats_train.shape[0]
        if name == "pca":
            from sklearn.decomposition import PCA
            out = PCA(n_components=2, random_state=seed).fit_transform(joint)
        elif name == "tsne":
            from sklearn.manifold import TSNE
            out = TSNE(n_components=2, random_state=seed,
                       init="pca", perplexity=30).fit_transform(joint)
        else:  # umap
            from umap import UMAP
            out = UMAP(n_components=2, random_state=seed).fit_transform(joint)
        return out[:split], out[split:]

    if name == "xfpad":
        import torch
        from xfpad.models import GeometricEncoder
        if geo_ckpt is None or not Path(geo_ckpt).exists():
            raise FileNotFoundError(
                f"xfpad representation needs a geometric checkpoint; "
                f"got {geo_ckpt}")
        model = GeometricEncoder(dropout=dropout)
        state = torch.load(geo_ckpt, map_location="cpu")["model"]
        model.load_state_dict(state)
        model.eval()

        def _proj(x):
            with torch.no_grad():
                t = torch.from_numpy(np.asarray(x)).float()
                return model(t).cpu().numpy()

        return _proj(feats_train), _proj(feats_test)

    raise ValueError(f"Unknown representation '{name}'.")


# ---------------------------------------------------------------------------
# One representation -> attribution results
# ---------------------------------------------------------------------------

def _attribute(z_train, y_train, train_names_d,
               z_test, y_test, test_names_d, tau):
    bf_train = bona_fide_label(train_names_d)
    bf_key = train_names_d[bf_train]
    prototype_order = [train_names_d[lbl]
                       for lbl in sorted(train_names_d) if lbl != bf_train]
    results = analyze_unseen_pais(
        features_train=np.asarray(z_train),
        labels_train=np.asarray(y_train),
        train_names=train_names_d,
        features_unseen=np.asarray(z_test),
        labels_unseen=np.asarray(y_test),
        unseen_names=test_names_d,
        prototype_order=prototype_order,
        bf_key=bf_key,
        tau=tau,
    )
    return results, prototype_order


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_table(rep: str, results: Dict[str, Dict], top_n: int = 4) -> None:
    print()
    print("=" * 78)
    print(f"REPRESENTATION: {rep}")
    print("=" * 78)
    header = f"{'Unseen PAI':<26} | top anchors (p_u,k)                     | H_norm"
    print(header)
    print("-" * len(header))
    for pai, res in results.items():
        ranked = res["ranked_anchors"][:top_n]
        anchors = ", ".join(f"{n} {w:.2f}" for n, w in ranked)
        print(f"{pai:<26} | {anchors:<38} | {res['entropy']:.2f}")


def _summary(all_results: Dict[str, Dict], reference: str = "xfpad") -> None:
    """Cross-representation summary: mean entropy + top-1 agreement vs reference."""
    print()
    print("#" * 78)
    print("SUMMARY  (lower H_norm = more decisive; agree = top-1 anchor == "
          f"{reference} top-1)")
    print("#" * 78)
    ref = all_results.get(reference)
    pais = list(next(iter(all_results.values())).keys())
    print(f"{'representation':<12} | {'mean H_norm':>11} | {'top1 agree vs '+reference:>22}")
    print("-" * 52)
    for rep, results in all_results.items():
        ents = [results[p]["entropy"] for p in pais]
        mean_h = float(np.mean(ents))
        if ref is None:
            agree = float("nan")
        else:
            hits = sum(1 for p in pais
                       if results[p]["ranked_anchors"][0][0]
                       == ref[p]["ranked_anchors"][0][0])
            agree = hits / len(pais)
        print(f"{rep:<12} | {mean_h:>11.3f} | {agree:>22.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = base_parser("Diagnostic-baseline attribution comparison.")
    parser.add_argument("--test-config", default=None,
                        help="Optional separate scanner YAML for the test "
                             "split (cross-sensor).")
    parser.add_argument("--reductions", nargs="+",
                        default=["raw", "pca", "tsne", "umap", "xfpad"],
                        help="Representations to compare.")
    parser.add_argument("--geo-ckpt", default=None,
                        help="Path to the trained g_psi checkpoint for the "
                             "'xfpad' representation. Defaults to "
                             "checkpoints/geometric_<scanner>_<seed>.pth.")
    parser.add_argument("--save-json", default=None)
    args = parser.parse_args()

    train_cfg = load_with_overrides(args)
    test_cfg = (train_cfg if args.test_config is None
                else load_config(args.test_config, base_yaml=args.base_config))
    seed = train_cfg.seed

    # Cached backbone features (extracted with the train-scanner backbone).
    feats_train = np.load(features_path(train_cfg.paths.features_dir,
                                        train_cfg.scanner, "train"))
    feats_test = np.load(features_path(test_cfg.paths.features_dir,
                                       test_cfg.scanner, "test"))

    train_paths = read_split(split_path(train_cfg.paths.splits_dir,
                                        train_cfg.scanner, "train"))
    test_paths = read_split(split_path(test_cfg.paths.splits_dir,
                                       test_cfg.scanner, "test"))
    y_train = build_labels(train_paths, known_pairs(train_cfg))
    y_test = build_labels(test_paths, unseen_pairs(test_cfg))

    train_names_d = known_names(train_cfg)
    test_names_d = unseen_names(test_cfg)

    geo_ckpt = (Path(args.geo_ckpt) if args.geo_ckpt
                else geometric_ckpt(train_cfg.paths.checkpoints,
                                    train_cfg.scanner, seed))

    LOG.info("train=%s (%d) test=%s (%d) seed=%d reductions=%s",
             train_cfg.scanner, feats_train.shape[0],
             test_cfg.scanner, feats_test.shape[0], seed, args.reductions)

    all_results: Dict[str, Dict] = {}
    for rep in args.reductions:
        LOG.info("computing representation '%s' ...", rep)
        z_train, z_test = _reduce(rep, feats_train, feats_test, seed,
                                  geo_ckpt, train_cfg.geometric.dropout)
        results, proto_order = _attribute(
            z_train, y_train, train_names_d,
            z_test, y_test, test_names_d, train_cfg.attribution.tau)
        all_results[rep] = results
        _print_table(rep, results)

    _summary(all_results, reference="xfpad" if "xfpad" in all_results
             else args.reductions[-1])

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        serial = {
            rep: {pai: {"ranked": [(n, float(w)) for n, w in r["ranked_anchors"]],
                        "entropy": float(r["entropy"])}
                  for pai, r in res.items()}
            for rep, res in all_results.items()
        }
        with out.open("w") as f:
            json.dump(serial, f, indent=2)
        LOG.info("saved -> %s", out)


if __name__ == "__main__":
    main()
