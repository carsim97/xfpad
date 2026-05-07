"""Optional — find an angularly coherent label assignment for known PAIs.

Given the 1280-D backbone embeddings of the training split, this script
computes the cosine-similarity matrix between PAI centroids and searches
for the cyclic ordering that maximises the sum of similarities between
adjacent prototypes. The resulting permutation can be used to relabel the
training PAIs so that geometrically similar materials land in adjacent
angular sectors of the X-FPAD manifold.

Run AFTER `phase1_train.py --stage features`. The script does not modify
any config; it prints a report and (optionally) writes a JSON file with
the new label assignment that you can transcribe into your config.

Usage example
-------------
    python scripts/optimize_ordering.py -c configs/greenbit.yaml \\
        --save-json outputs/ordering_greenbit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import (  # noqa: E402
    base_parser,
    bona_fide_label,
    known_names,
    known_pairs,
    load_with_overrides,
)
from xfpad.data import build_labels  # noqa: E402
from xfpad.training import (  # noqa: E402
    canonical_circular_form,
    compute_pai_centroids,
    cosine_similarity_matrix,
    cumulative_circular_similarity,
    find_optimal_circular_ordering,
    format_ordering_report,
)
from xfpad.utils import (  # noqa: E402
    ensure_dir,
    features_path,
    get_logger,
    read_split,
    split_path,
)

LOG = get_logger("optimize_ordering")


def main() -> None:
    parser = base_parser("Optimise circular ordering of PAI prototypes.")
    parser.add_argument("--save-json", default=None,
                        help="If given, write the ordering result as JSON.")
    args = parser.parse_args()
    cfg = load_with_overrides(args)

    fpath = features_path(cfg.paths.features_dir, cfg.scanner, "train")
    if not fpath.exists():
        raise FileNotFoundError(
            f"Train features missing: {fpath}. Run Phase 1 'features' first."
        )
    features = np.load(fpath)

    paths = read_split(split_path(cfg.paths.splits_dir, cfg.scanner, "train"))
    pairs = known_pairs(cfg)
    labels = np.array(build_labels(paths, pairs))

    LOG.info("Features %s, labels %s", features.shape, labels.shape)

    centroids, pai_labels = compute_pai_centroids(features, labels)
    K = len(pai_labels)

    names_map = known_names(cfg)
    bf_lbl = bona_fide_label(names_map)
    class_names = [names_map[int(lbl)] for lbl in pai_labels if lbl != bf_lbl]
    LOG.info("Detected %d PAI classes: %s", K, class_names)

    S = cosine_similarity_matrix(centroids)
    order, score, method = find_optimal_circular_ordering(S)

    canonical = list(canonical_circular_form(order))
    if cumulative_circular_similarity(canonical, S) >= score - 1e-9:
        order = canonical

    report = format_ordering_report(
        order=order, S=S,
        current_labels=pai_labels,
        class_names=class_names,
        score=score, method=method,
    )
    print(report)

    if args.save_json:
        out = Path(args.save_json)
        ensure_dir(out.parent)
        with out.open("w") as f:
            json.dump({
                "method": method,
                "score": float(score),
                "K": int(K),
                "order_indices": list(order),
                "old_labels_in_order": [int(pai_labels[i]) for i in order],
                "new_labels_in_order": list(range(1, K + 1)),
                "material_names_in_order": [class_names[i] for i in order],
                "similarity_matrix": S.tolist(),
            }, f, indent=2)
        LOG.info("Saved ordering -> %s", out)


if __name__ == "__main__":
    main()
