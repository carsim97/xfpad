"""Phase 2 — Directional mapping of unseen PAIs onto known prototypes.

For one or more seeds, this script:
  1. Loads the trained backbone f_phi and geometric encoder g_psi.
  2. Projects training and unseen-PAI samples into the 2-D manifold.
  3. Computes the soft attribution weights p_{u,k} (Eq. 5) for each
     unseen PAI against the K training prototypes.
  4. Optionally renders the latent-space figures.
  5. Aggregates over seeds (mean +/- std) and emits a Phase 2 table.

Usage examples
--------------
    # Single seed, with plot.
    python scripts/phase2_map_unseen.py -c configs/greenbit.yaml --plot

    # Multi-seed, paper-style aggregation.
    python scripts/phase2_map_unseen.py -c configs/dermalog.yaml --num-runs 10 \\
                                        --save-json outputs/phase2_dermalog.json

    # Cross-sensor (train on greenbit, evaluate on dermalog test split).
    python scripts/phase2_map_unseen.py -c configs/greenbit.yaml \\
                                        --test-config configs/dermalog.yaml
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import (  # noqa: E402
    axes_lim,
    base_parser,
    bona_fide_label,
    known_names,
    known_pairs,
    load_with_overrides,
    unseen_names,
    unseen_pairs,
)
from xfpad.config import Config, load_config  # noqa: E402
from xfpad.data import FingerprintDataset, build_labels  # noqa: E402
from xfpad.metrics import (  # noqa: E402
    analyze_unseen_pais,
    calculate_metrics,
)
from xfpad.models import FeatureExtractor, GeometricEncoder  # noqa: E402
from xfpad.utils import (  # noqa: E402
    ensure_dir,
    feature_extractor_ckpt,
    geometric_ckpt,
    get_logger,
    read_split,
    resolve_device,
    set_seed,
    split_path,
)
from xfpad.viz import plot_latent_space  # noqa: E402

LOG = get_logger("phase2")


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _project(paths: List[str],
             labels: List[int],
             fe: FeatureExtractor,
             ge: GeometricEncoder,
             device: torch.device,
             num_workers: int) -> np.ndarray:
    """Run f_phi then g_psi over a list of image paths."""
    dataset = FingerprintDataset(paths, labels)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    out: List[np.ndarray] = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="projecting", leave=False):
            x = x.to(device)
            feats = fe(x)
            z = ge(feats)
            out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)


def _load_models(train_cfg: Config,
                 seed: int,
                 device: torch.device) -> Tuple[FeatureExtractor, GeometricEncoder]:
    fe_ckpt = feature_extractor_ckpt(train_cfg.paths.checkpoints, train_cfg.scanner)
    ge_ckpt = geometric_ckpt(train_cfg.paths.checkpoints, train_cfg.scanner, seed)
    if not fe_ckpt.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {fe_ckpt}")
    if not ge_ckpt.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {ge_ckpt}")

    fe = FeatureExtractor(
        in_channels=train_cfg.backbone.in_channels,
        training_mode=False,
    ).to(device)
    fe.load_state_dict(torch.load(fe_ckpt, map_location=device)["model"], strict=False)
    fe.eval()

    ge = GeometricEncoder(dropout=train_cfg.geometric.dropout).to(device)
    ge.load_state_dict(torch.load(ge_ckpt, map_location=device)["model"])
    ge.eval()

    return fe, ge


def _radial_zones(z: np.ndarray, labels: np.ndarray,
                  rho_bf: float, rho_pa: float) -> Dict[str, float]:
    """Where do the PAI samples sit radially? (neutral-zone diagnostic)

    The manifold assigns bona fide to |z| <= rho_bf and PAIs to |z| >= rho_pa;
    the band in between carries no decision. Under cross-sensor domain shift the
    unseen clusters are expected to collapse into that band, which is precisely
    what has to be quantified rather than asserted.
    """
    mask = labels > 0
    if not np.any(mask):
        return {"n": 0, "below_rho_bf": float("nan"), "neutral": float("nan"),
                "above_rho_pa": float("nan"), "median_radius": float("nan")}
    r = np.linalg.norm(z[mask], axis=1)
    return {
        "n": int(mask.sum()),
        "below_rho_bf": float(np.mean(r <= rho_bf)),
        "neutral": float(np.mean((r > rho_bf) & (r < rho_pa))),
        "above_rho_pa": float(np.mean(r >= rho_pa)),
        "median_radius": float(np.median(r)),
    }


# ---------------------------------------------------------------------------
# Per-seed analysis
# ---------------------------------------------------------------------------

def _run_one_seed(train_cfg: Config,
                  test_cfg: Config,
                  seed: int,
                  do_plot: bool,
                  outputs_root: Path) -> Dict[str, Dict]:
    device = resolve_device(train_cfg.device)
    set_seed(seed)

    fe, ge = _load_models(train_cfg, seed, device)

    train_paths = read_split(split_path(train_cfg.paths.splits_dir, train_cfg.scanner, "train"))
    train_labels = build_labels(train_paths, known_pairs(train_cfg))
    z_train = _project(train_paths, train_labels, fe, ge, device,
                       train_cfg.backbone.num_workers)

    # Unseen side (possibly different scanner).
    test_paths = read_split(split_path(test_cfg.paths.splits_dir, test_cfg.scanner, "test"))
    test_labels = build_labels(test_paths, unseen_pairs(test_cfg))
    z_test = _project(test_paths, test_labels, fe, ge, device,
                      train_cfg.backbone.num_workers)

    train_labels_arr = np.asarray(train_labels)
    test_labels_arr = np.asarray(test_labels)

    # Diagnostics on the training manifold. These describe how g_psi organised
    # its OWN training data and are therefore invariant to --test-config: in a
    # cross-sensor run they are identical to the intra-sensor ones by
    # construction, so they must NOT be quoted as cross-sensor evidence.
    # NOTE: calculate_metrics returns numpy float32; json.dump streams its
    # output and dies partway through on a non-serialisable value, leaving a
    # TRUNCATED file behind. Cast at the source.
    bfo, rci, acs = (float(v) for v in calculate_metrics(
        z_train, train_labels_arr, rho_bf=train_cfg.loss.rho_bf))

    # Diagnostics on the projected TEST data — this is the side that actually
    # changes under --test-config, and the one the neutral-zone diagnostic reads.
    bfo_t, rci_t, acs_t = (float(v) for v in calculate_metrics(
        z_test, test_labels_arr, rho_bf=train_cfg.loss.rho_bf))

    rho_bf = train_cfg.loss.rho_bf
    rho_pa = rho_bf + train_cfg.loss.delta_rho
    zones = _radial_zones(z_test, test_labels_arr, rho_bf, rho_pa)
    zones_per_pai = {
        name: _radial_zones(z_test, np.where(test_labels_arr == lbl, 1, 0),
                            rho_bf, rho_pa)
        for lbl, name in unseen_names(test_cfg).items()
        if lbl != 0 and np.any(test_labels_arr == lbl)
    }

    LOG.info("seed=%d  train-side BFO=%.2f RCI=%.2f ACS=%.2f | "
             "test-side BFO=%.2f RCI=%.2f ACS=%.2f | neutral zone=%.1f%%",
             seed, bfo, rci, acs, bfo_t, rci_t, acs_t, 100 * zones["neutral"])

    # Soft attribution.
    train_names_d = known_names(train_cfg)
    test_names_d = unseen_names(test_cfg)
    bf_train = bona_fide_label(train_names_d)
    bf_key = train_names_d[bf_train]

    prototype_order = [
        train_names_d[lbl]
        for lbl in sorted(train_names_d) if lbl != bf_train
    ]

    results = analyze_unseen_pais(
        features_train=z_train,
        labels_train=train_labels_arr,
        train_names=train_names_d,
        features_unseen=z_test,
        labels_unseen=test_labels_arr,
        unseen_names=test_names_d,
        prototype_order=prototype_order,
        bf_key=bf_key,
        tau=train_cfg.attribution.tau,
    )

    # Persist projections for downstream use / debugging. The directory must
    # name BOTH sides: with only the train scanner, a cross-sensor run silently
    # overwrites the intra-sensor projections it should be compared against.
    tag = (train_cfg.scanner if test_cfg.scanner == train_cfg.scanner
           else f"{train_cfg.scanner}_to_{test_cfg.scanner}")
    proj_dir = ensure_dir(outputs_root / "projections" / tag)
    np.savez(
        proj_dir / f"seed{seed}.npz",
        z_train=z_train, labels_train=train_labels_arr,
        z_test=z_test, labels_test=test_labels_arr,
    )

    if do_plot:
        # Same tag as the projections: without it a cross-sensor run overwrites
        # the intra-sensor figures it is meant to be compared against.
        plot_root = ensure_dir(outputs_root / "plots" / tag / f"seed{seed}")
        T = train_cfg.loss.rho_bf ** 2
        derived_lim = plot_latent_space(
            z_train, train_labels_arr, train_names_d,
            plot_root / "training.png",
            mode="training",
            T=T,
            axes_lim=None,
            figsize=tuple(train_cfg.plot.figsize),
            dpi=train_cfg.plot.dpi,
            alpha=train_cfg.plot.alpha,
            bf_color=train_cfg.plot.bf_color,
            unseen_color=train_cfg.plot.unseen_color,
        )
        plot_latent_space(
            z_test, test_labels_arr, test_names_d,
            plot_root / "unseen.png",
            mode="unseen",
            T=T,
            axes_lim=derived_lim,
            figsize=tuple(train_cfg.plot.figsize),
            dpi=train_cfg.plot.dpi,
            alpha=train_cfg.plot.alpha,
            bf_color=train_cfg.plot.bf_color,
            unseen_color=train_cfg.plot.unseen_color,
        )
        LOG.info("seed=%d plots -> %s", seed, plot_root)

    return {
        "seed": seed,
        "train_scanner": train_cfg.scanner,
        "test_scanner": test_cfg.scanner,
        "metrics": {"BFO": bfo, "RCI": rci, "ACS": acs},
        "metrics_test": {"BFO": bfo_t, "RCI": rci_t, "ACS": acs_t},
        "zones": zones,
        "zones_per_pai": zones_per_pai,
        "rho_bf": rho_bf,
        "rho_pa": rho_pa,
        "attribution": {
            pai: {
                "weights": {n: float(w) for n, w in zip(prototype_order, res["weights"])},
                "ranked": [(n, float(w)) for n, w in res["ranked_anchors"]],
                "entropy": float(res["entropy"]),
            }
            for pai, res in results.items()
        },
        "prototype_order": prototype_order,
    }


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def _aggregate(per_seed: List[Dict]) -> Dict:
    """Mean ± std of attribution weights and metrics across seeds."""
    proto_order = per_seed[0]["prototype_order"]

    pai_names = list(per_seed[0]["attribution"].keys())
    aggr_attr: Dict[str, Dict] = {}
    for pai in pai_names:
        stacked = np.stack([
            np.array([s["attribution"][pai]["weights"][n] for n in proto_order])
            for s in per_seed
        ], axis=0)  # (S, K)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        ranked = sorted(
            zip(proto_order, mean, std),
            key=lambda t: t[1], reverse=True,
        )
        aggr_attr[pai] = {
            "mean":    {n: float(m) for n, m in zip(proto_order, mean)},
            "std":     {n: float(s) for n, s in zip(proto_order, std)},
            "ranked":  [(n, float(m), float(s)) for n, m, s in ranked],
        }

    def _ms(getter) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for k in getter(per_seed[0]):
            vals = np.array([float(getter(s)[k]) for s in per_seed])
            out[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
        return out

    zone_keys = [k for k in per_seed[0]["zones"] if k != "n"]
    zones_per_pai = {
        pai: {k: {"mean": float(np.mean([s["zones_per_pai"][pai][k] for s in per_seed])),
                  "std": float(np.std([s["zones_per_pai"][pai][k] for s in per_seed]))}
              for k in zone_keys}
        for pai in per_seed[0]["zones_per_pai"]
    }

    return {
        "metrics": _ms(lambda s: s["metrics"]),
        "metrics_test": _ms(lambda s: s["metrics_test"]),
        "zones": _ms(lambda s: {k: s["zones"][k] for k in zone_keys}),
        "zones_per_pai": zones_per_pai,
        "attribution": aggr_attr,
        "prototype_order": proto_order,
        "n_seeds": len(per_seed),
        "train_scanner": per_seed[0]["train_scanner"],
        "test_scanner": per_seed[0]["test_scanner"],
        "rho_bf": per_seed[0]["rho_bf"],
        "rho_pa": per_seed[0]["rho_pa"],
    }


def _format_aggregated_table(aggr: Dict, top_n: int = 4, tail_threshold: float = 0.07) -> str:
    """Format aggregated Phase 2 results as Table III of the paper."""
    lines = []
    header = (f"{'Unseen PAI':<28}  Top anchors (mean ± std), "
              f"tail = sum of weights below {tail_threshold:.2f}")
    lines.append(header)
    lines.append("-" * len(header))
    for pai, data in aggr["attribution"].items():
        ranked = data["ranked"]
        top = ranked[:top_n]
        tail = ranked[top_n:]
        tail_sum = sum(w for _, w, _ in tail if w < tail_threshold)
        top_str = ", ".join(f"{n} ({m:.2f}±{s:.2f})" for n, m, s in top)
        lines.append(f"{pai:<28}  {top_str}   tail={tail_sum:.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = base_parser("Phase 2 - Directional mapping of unseen PAIs.")
    parser.add_argument(
        "--test-config", default=None,
        help="Optional separate scanner YAML for the test split (cross-sensor). "
             "If omitted, the same config as --config is used (intra-sensor).",
    )
    parser.add_argument(
        "--num-runs", type=int, default=1,
        help="Number of seeds to aggregate (uses cfg.seed, cfg.seed+1, ...).",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Render latent-space figures for each seed.",
    )
    parser.add_argument(
        "--save-json", default=None,
        help="If given, write the aggregated Phase 2 results as JSON.",
    )
    args = parser.parse_args()

    train_cfg = load_with_overrides(args)
    if args.test_config is None:
        test_cfg = train_cfg
    else:
        # The test config is loaded layered on the same base_config.
        test_cfg = load_config(args.test_config, base_yaml=args.base_config)

    LOG.info("=== Phase 2 / train=%s test=%s seeds=%d ===",
             train_cfg.scanner, test_cfg.scanner, args.num_runs)

    outputs_root = ensure_dir(Path(train_cfg.paths.outputs))
    per_seed: List[Dict] = []
    for k in range(args.num_runs):
        seed = train_cfg.seed + k
        per_seed.append(_run_one_seed(train_cfg, test_cfg, seed,
                                      do_plot=args.plot,
                                      outputs_root=outputs_root))

    aggr = _aggregate(per_seed)

    print()
    print("=" * 78)
    print(f"PHASE 2 — train={train_cfg.scanner}  test={test_cfg.scanner}  "
          f"n_seeds={aggr['n_seeds']}")
    print("=" * 78)
    m, mt, zo = aggr["metrics"], aggr["metrics_test"], aggr["zones"]
    print(f"train-side geometry  BFO={m['BFO']['mean']:.2f}±{m['BFO']['std']:.2f}  "
          f"RCI={m['RCI']['mean']:.2f}±{m['RCI']['std']:.2f}  "
          f"ACS={m['ACS']['mean']:.2f}±{m['ACS']['std']:.2f}"
          f"   (invariant to --test-config)")
    print(f"test-side  geometry  BFO={mt['BFO']['mean']:.2f}±{mt['BFO']['std']:.2f}  "
          f"RCI={mt['RCI']['mean']:.2f}±{mt['RCI']['std']:.2f}  "
          f"ACS={mt['ACS']['mean']:.2f}±{mt['ACS']['std']:.2f}")
    print(f"\nradial placement of unseen PAI samples "
          f"(rho_bf={aggr['rho_bf']:.1f}, rho_pa={aggr['rho_pa']:.1f}):")
    print(f"   inside bona fide core (<=rho_bf) {100*zo['below_rho_bf']['mean']:5.1f}%"
          f" ±{100*zo['below_rho_bf']['std']:.1f}")
    print(f"   NEUTRAL ZONE                     {100*zo['neutral']['mean']:5.1f}%"
          f" ±{100*zo['neutral']['std']:.1f}")
    print(f"   attack region (>=rho_pa)         {100*zo['above_rho_pa']['mean']:5.1f}%"
          f" ±{100*zo['above_rho_pa']['std']:.1f}")
    print(f"   median radius                    {zo['median_radius']['mean']:5.2f}"
          f" ±{zo['median_radius']['std']:.2f}")
    print(f"\n{'unseen PAI':<28} {'core%':>7} {'neutral%':>10} {'attack%':>9} {'med r':>7}")
    for pai, z in aggr["zones_per_pai"].items():
        print(f"{pai:<28} {100*z['below_rho_bf']['mean']:>7.1f} "
              f"{100*z['neutral']['mean']:>10.1f} {100*z['above_rho_pa']['mean']:>9.1f} "
              f"{z['median_radius']['mean']:>7.2f}")
    print()
    print(_format_aggregated_table(aggr))

    if args.save_json:
        out = Path(args.save_json)
        ensure_dir(out.parent)
        with out.open("w") as f:
            json.dump({"per_seed": per_seed, "aggregated": aggr}, f, indent=2)
        LOG.info("Saved Phase 2 JSON -> %s", out)


if __name__ == "__main__":
    main()
