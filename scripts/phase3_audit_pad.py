"""Phase 3 — Cross-Model Feature Consistency via Targeted Ablation.

Trains or evaluates a binary PAD detector (one of the three audited
backbones: MobileNet-v2, ResNet-18, DenseNet-121) under either the
baseline training set or an ablated variant in which all paths
containing one or more substrings are removed.

Usage examples
--------------
    # Train 10 seeds of MobileNet-v2 baseline on Green Bit:
    python scripts/phase3_audit_pad.py -c configs/greenbit.yaml \\
        --backbone mobilenet_v2 --action train --num-runs 10

    # Train an ablation removing 'Wood' substrings:
    python scripts/phase3_audit_pad.py -c configs/greenbit.yaml \\
        --backbone resnet18 --action train --num-runs 10 \\
        --ablate Wood --ablation-name without_wood_glue

    # Evaluate each trained seed on the unseen-PAI split:
    python scripts/phase3_audit_pad.py -c configs/greenbit.yaml \\
        --backbone mobilenet_v2 --action eval --num-runs 10 \\
        --ablation-name baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import (  # noqa: E402
    base_parser,
    bona_fide_label,
    known_names,
    known_pairs,
    load_with_overrides,
)
from xfpad.config import Config  # noqa: E402
from xfpad.data import FingerprintDataset  # noqa: E402
from xfpad.metrics import apcer_bpcer, apcer_per_unseen_pai  # noqa: E402
from xfpad.models import PadDetector  # noqa: E402
from xfpad.utils import (  # noqa: E402
    ensure_dir,
    get_logger,
    pad_detector_ckpt,
    read_split,
    resolve_device,
    set_seed,
    split_path,
)

LOG = get_logger("phase3")


# ---------------------------------------------------------------------------
# Binary label assignment for the PAD detector (1 = live, 0 = spoof)
# ---------------------------------------------------------------------------

def _binary_label_for_path(path: str) -> int:
    """1 if 'live' (case-insensitive) is in the path, else 0."""
    return 1 if "live" in path.lower() else 0


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------

def _apply_ablation(paths: List[str], ablate: List[str]) -> List[str]:
    """Remove paths whose name contains any of the ablation substrings."""
    if not ablate:
        return list(paths)
    keep = [p for p in paths if not any(s in p for s in ablate)]
    LOG.info("Ablation substrings %s: %d / %d kept",
             ablate, len(keep), len(paths))
    return keep


def _default_ablation_name(ablate: List[str] | None) -> str:
    if not ablate:
        return "baseline"
    return "without_" + "_".join(s.lower() for s in ablate)


# ---------------------------------------------------------------------------
# Train one seed
# ---------------------------------------------------------------------------

def _train_one(cfg: Config,
               backbone: str,
               run_idx: int,
               ablate: List[str],
               ablation_name: str) -> Path:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed + run_idx)

    train_paths = read_split(split_path(cfg.paths.splits_dir, cfg.scanner, "train"))
    train_paths = _apply_ablation(train_paths, ablate)
    labels = [_binary_label_for_path(p) for p in train_paths]

    n_live = sum(labels)
    n_spoof = len(labels) - n_live
    LOG.info("[%s/%s/run %d] train: %d samples (%d live, %d spoof)",
             cfg.scanner, ablation_name, run_idx, len(labels), n_live, n_spoof)

    dataset = FingerprintDataset(train_paths, labels)
    loader = DataLoader(
        dataset,
        batch_size=cfg.pad_detector.batch_size,
        shuffle=True,
        num_workers=cfg.pad_detector.num_workers,
        pin_memory=True,
    )

    model = PadDetector(backbone=backbone,
                        in_channels=cfg.backbone.in_channels).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.pad_detector.lr,
        weight_decay=cfg.pad_detector.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.pad_detector.num_epochs,
    )

    for epoch in range(cfg.pad_detector.num_epochs):
        model.train()
        running = 0.0
        for x, y in tqdm(loader, desc=f"epoch {epoch + 1}", leave=False):
            x = x.to(device)
            y = y.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            LOG.info("[%s/%s/run %d] epoch %d/%d loss=%.4f",
                     cfg.scanner, ablation_name, run_idx,
                     epoch + 1, cfg.pad_detector.num_epochs, running)

    out = pad_detector_ckpt(cfg.paths.checkpoints, cfg.scanner,
                            backbone, ablation_name, run_idx)
    ensure_dir(out.parent)
    torch.save({
        "model": model.state_dict(),
        "backbone": backbone,
        "ablation": ablation_name,
        "ablate_substrings": ablate,
        "run": run_idx,
    }, out)
    LOG.info("Saved -> %s", out)
    return out


# ---------------------------------------------------------------------------
# Evaluate one seed
# ---------------------------------------------------------------------------

def _eval_one(cfg: Config,
              backbone: str,
              run_idx: int,
              ablation_name: str) -> Dict:
    device = resolve_device(cfg.device)

    ckpt = pad_detector_ckpt(cfg.paths.checkpoints, cfg.scanner,
                             backbone, ablation_name, run_idx)
    if not ckpt.exists():
        raise FileNotFoundError(f"PAD checkpoint not found: {ckpt}")

    model = PadDetector(backbone=backbone,
                        in_channels=cfg.backbone.in_channels).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()

    test_paths = read_split(split_path(cfg.paths.splits_dir, cfg.scanner, "test"))
    labels = [_binary_label_for_path(p) for p in test_paths]

    dataset = FingerprintDataset(test_paths, labels)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=cfg.pad_detector.num_workers, pin_memory=True,
    )

    preds: List[int] = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc=f"eval run {run_idx}", leave=False):
            logits = model(x.to(device))
            p = torch.sigmoid(logits).item()
            preds.append(1 if p > cfg.pad_detector.threshold else 0)

    preds = np.array(preds)
    apcer, bpcer, ace = apcer_bpcer(np.array(labels), preds)

    # Per-PAI APCER (substring matching from cfg.unseen_labels.mapping).
    pai_defs = _per_pai_definitions(cfg)
    per_pai = apcer_per_unseen_pai(test_paths, preds.tolist(), pai_defs)

    LOG.info("[%s/%s/%s/run %d] APCER=%.2f BPCER=%.2f ACE=%.2f",
             cfg.scanner, backbone, ablation_name, run_idx, apcer, bpcer, ace)
    for k, v in per_pai.items():
        LOG.info("    %-30s APCER=%.2f", k, v)

    return {
        "run": run_idx,
        "backbone": backbone,
        "ablation": ablation_name,
        "APCER": apcer,
        "BPCER": bpcer,
        "ACE": ace,
        "per_pai_apcer": per_pai,
    }


def _per_pai_definitions(cfg: Config) -> Dict[str, List[str]]:
    """Pull substring lists from cfg.unseen_labels.mapping (skipping bona fide)."""
    bf_label = bona_fide_label({int(k): v for k, v in cfg.unseen_labels.names.items()})
    out: Dict[str, List[str]] = {}
    for entry in cfg.unseen_labels.mapping:
        lbl = int(entry["label"])
        if lbl == bf_label:
            continue
        name = cfg.unseen_labels.names[lbl] if lbl in cfg.unseen_labels.names else cfg.unseen_labels.names[str(lbl)]
        out[name] = list(entry["match"])
    return out


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------

def _aggregate_eval(results: List[Dict]) -> Dict:
    if not results:
        return {}
    # Overall APCER/BPCER/ACE.
    metric_keys = ["APCER", "BPCER", "ACE"]
    overall = {k: {"mean": float(np.mean([r[k] for r in results])),
                   "std":  float(np.std([r[k] for r in results]))}
               for k in metric_keys}
    # Per-PAI APCER.
    pai_keys = list(results[0]["per_pai_apcer"].keys())
    per_pai: Dict[str, Dict[str, float]] = {}
    for k in pai_keys:
        vals = np.array([r["per_pai_apcer"][k] for r in results], dtype=float)
        vals = vals[~np.isnan(vals)]
        per_pai[k] = {
            "mean": float(vals.mean()) if vals.size else float("nan"),
            "std":  float(vals.std())  if vals.size else float("nan"),
            "n":    int(vals.size),
        }
    return {"overall": overall, "per_pai": per_pai, "n_runs": len(results)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = base_parser("Phase 3 - Audited PAD detector training / evaluation.")
    parser.add_argument(
        "--backbone", required=True,
        choices=list(PadDetector.SUPPORTED),
        help="Audited backbone.",
    )
    parser.add_argument(
        "--action", required=True, choices=["train", "eval", "both"],
        help="Train, evaluate, or both.",
    )
    parser.add_argument(
        "--num-runs", type=int, default=10,
        help="Number of seeds (default 10, matching the paper).",
    )
    parser.add_argument(
        "--ablate", nargs="*", default=[],
        help="Substrings to remove from the training paths "
             "(e.g. --ablate Wood). Multiple substrings act as OR.",
    )
    parser.add_argument(
        "--ablation-name", default=None,
        help="Friendly name for this ablation. Auto-derived from --ablate "
             "if omitted (e.g. 'without_wood', 'baseline').",
    )
    parser.add_argument(
        "--save-json", default=None,
        help="Write aggregated eval results to JSON.",
    )
    args = parser.parse_args()

    cfg = load_with_overrides(args)
    name = args.ablation_name or _default_ablation_name(args.ablate)

    LOG.info("=== Phase 3 / scanner=%s backbone=%s ablation=%s n_runs=%d ===",
             cfg.scanner, args.backbone, name, args.num_runs)

    if args.action in {"train", "both"}:
        for k in range(args.num_runs):
            _train_one(cfg, args.backbone, run_idx=k,
                       ablate=args.ablate, ablation_name=name)

    if args.action in {"eval", "both"}:
        per_seed: List[Dict] = []
        for k in range(args.num_runs):
            per_seed.append(_eval_one(cfg, args.backbone, run_idx=k,
                                      ablation_name=name))
        aggr = _aggregate_eval(per_seed)

        print()
        print("=" * 78)
        print(f"PHASE 3 EVAL  scanner={cfg.scanner}  backbone={args.backbone}  "
              f"ablation={name}  n_runs={aggr['n_runs']}")
        print("=" * 78)
        ov = aggr["overall"]
        print(f"  APCER {ov['APCER']['mean']:.2f}±{ov['APCER']['std']:.2f}  "
              f"BPCER {ov['BPCER']['mean']:.2f}±{ov['BPCER']['std']:.2f}  "
              f"ACE {ov['ACE']['mean']:.2f}±{ov['ACE']['std']:.2f}")
        for k, v in aggr["per_pai"].items():
            print(f"  {k:<30}  {v['mean']:.2f}±{v['std']:.2f}  (n={v['n']})")

        if args.save_json:
            out = Path(args.save_json)
            ensure_dir(out.parent)
            with out.open("w") as f:
                json.dump({"per_seed": per_seed, "aggregated": aggr}, f, indent=2)
            LOG.info("Saved Phase 3 JSON -> %s", out)


if __name__ == "__main__":
    main()
