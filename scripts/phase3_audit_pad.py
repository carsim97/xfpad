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

import json
import random
import sys
from collections import Counter
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
from xfpad.data import FingerprintDataset, build_labels  # noqa: E402
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


def _apply_random_ablation(paths: List[str], n: int,
                           rng: random.Random,
                           cfg: Config | None = None) -> List[str]:
    """Matched-size random-removal control: drop n spoof paths
    chosen uniformly at random from the full spoof pool. The draw is
    governed by the per-run rng, so each seed sees an independent draw.

    When cfg is given, additionally log how many of the dropped samples
    belong to each material, so the supplementary can show the random draw
    did not accidentally deplete any single anchor, which is how a fully-random
    -- not stratified -- draw could end up mimicking a leave-one-material-out.
    The draw itself is unchanged, so results stay identical to before."""
    spoof_idx = [i for i, p in enumerate(paths)
                 if _binary_label_for_path(p) == 0]
    if n > len(spoof_idx):
        raise ValueError(f"--ablate-random-n {n} exceeds spoof pool "
                         f"({len(spoof_idx)}).")
    drop = set(rng.sample(spoof_idx, n))
    keep = [p for i, p in enumerate(paths) if i not in drop]
    LOG.info("Random ablation: dropped %d random spoof samples, %d / %d kept",
             n, len(keep), len(paths))
    if cfg is not None:
        names = known_names(cfg)
        labels = build_labels(paths, known_pairs(cfg))
        pool_by_mat = Counter(labels[i] for i in spoof_idx)
        drop_by_mat = Counter(labels[i] for i in drop)
        parts = [
            f"{names.get(lab, lab)}={drop_by_mat.get(lab, 0)}/{tot} "
            f"({drop_by_mat.get(lab, 0) / tot * 100:.1f}%)"
            for lab, tot in sorted(pool_by_mat.items())
        ]
        LOG.info("Random ablation composition of dropped set: %s",
                 ", ".join(parts))
    return keep


def _apply_subset_ablation(paths: List[str], substr: str, n: int,
                           rng: random.Random) -> List[str]:
    """Non-anchor matched control: drop n randomly chosen spoof
    paths belonging to ONE material (substring match), leaving the rest of
    that material in place. Isolates geometry from sample-count effects."""
    match_idx = [i for i, p in enumerate(paths)
                 if substr in p and _binary_label_for_path(p) == 0]
    if n > len(match_idx):
        raise ValueError(f"--subset-n {n} exceeds '{substr}' pool "
                         f"({len(match_idx)}).")
    drop = set(rng.sample(match_idx, n))
    keep = [p for i, p in enumerate(paths) if i not in drop]
    LOG.info("Subset ablation '%s': dropped %d of %d matching spoof samples, "
             "%d / %d kept", substr, n, len(match_idx), len(keep), len(paths))
    return keep


def _run_already_done(cfg: Config, backbone: str, ablation_name: str,
                      run_idx: int) -> bool:
    """True if this seed's checkpoint exists AND is loadable.

    Lets an interrupted cell resume at the seed it stopped on instead of
    restarting the whole cell. A checkpoint truncated mid-write fails to load
    and is simply retrained.
    """
    path = pad_detector_ckpt(cfg.paths.checkpoints, cfg.scanner,
                             backbone, ablation_name, run_idx)
    if not path.exists():
        return False
    try:
        torch.load(path, map_location="cpu")
        return True
    except Exception as e:
        LOG.warning("Checkpoint %s unreadable (%s); it will be retrained.",
                    path.name, e)
        return False


def _default_ablation_name(ablate: List[str] | None,
                           random_n: int | None = None,
                           subset_of: str | None = None,
                           subset_n: int | None = None) -> str:
    if ablate:
        return "without_" + "_".join(s.lower() for s in ablate)
    if random_n:
        return f"random_n{random_n}"
    if subset_of:
        return f"subset_{subset_of.lower()}_n{subset_n}"
    return "baseline"


# ---------------------------------------------------------------------------
# Train one seed
# ---------------------------------------------------------------------------

def _train_one(cfg: Config,
               backbone: str,
               run_idx: int,
               ablate: List[str],
               ablation_name: str,
               random_n: int | None = None,
               subset_of: str | None = None,
               subset_n: int | None = None) -> Path:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed + run_idx)

    train_paths = read_split(split_path(cfg.paths.splits_dir, cfg.scanner, "train"))
    train_paths = _apply_ablation(train_paths, ablate)
    if random_n:
        rng = random.Random(cfg.seed + run_idx)
        train_paths = _apply_random_ablation(train_paths, random_n, rng, cfg)
    if subset_of:
        rng = random.Random(cfg.seed + run_idx)
        train_paths = _apply_subset_ablation(train_paths, subset_of,
                                             subset_n or 0, rng)
    labels = [_binary_label_for_path(p) for p in train_paths]

    n_live = sum(labels)
    n_spoof = len(labels) - n_live
    LOG.info("[%s/%s/run %d] train: %d samples (%d live, %d spoof)",
             cfg.scanner, ablation_name, run_idx, len(labels), n_live, n_spoof)

    # cache_in_ram: decode each patch once instead of once per epoch. The
    # tensors are bit-identical; it only removes the CPU decode cost, which
    # serialises with training when num_workers=0.
    dataset = FingerprintDataset(train_paths, labels, cache_in_ram=True)
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
        "random_n": random_n,
        "subset_of": subset_of,
        "subset_n": subset_n,
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
              ablation_name: str,
              eval_batch_size: int = 64) -> Dict:
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

    dataset = FingerprintDataset(test_paths, labels, cache_in_ram=True)
    loader = DataLoader(
        dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=cfg.pad_detector.num_workers, pin_memory=True,
    )

    preds: List[int] = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc=f"eval run {run_idx}", leave=False):
            probs = torch.sigmoid(model(x.to(device))).squeeze(1)
            preds.extend((probs > cfg.pad_detector.threshold)
                         .long().cpu().tolist())

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
        "--ablate-random-n", type=int, default=None,
        help="Matched-size random control: remove N spoof samples chosen "
             "uniformly at random (independent draw per seed).",
    )
    parser.add_argument(
        "--ablate-subset-of", default=None,
        help="Non-anchor matched control: remove --subset-n random samples "
             "belonging to ONE material (substring match).",
    )
    parser.add_argument(
        "--subset-n", type=int, default=None,
        help="How many samples of --ablate-subset-of to remove.",
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=64,
        help="Batch size for evaluation (default 64; results identical to "
             "the paper's batch_size=1, only faster).",
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
    parser.add_argument(
        "--retrain-all", action="store_true",
        help="Retrain every seed even if its checkpoint already exists. By "
             "default a run whose checkpoint is present and loadable is "
             "skipped, so an interrupted cell resumes at the seed it stopped "
             "on instead of restarting from seed 0.",
    )
    args = parser.parse_args()

    if args.ablate_subset_of and not args.subset_n:
        parser.error("--ablate-subset-of requires --subset-n.")
    if args.ablate and (args.ablate_random_n or args.ablate_subset_of):
        parser.error("--ablate is mutually exclusive with the random/subset "
                     "controls; run them as separate ablation configs.")

    cfg = load_with_overrides(args)
    name = args.ablation_name or _default_ablation_name(
        args.ablate, args.ablate_random_n, args.ablate_subset_of, args.subset_n)

    LOG.info("=== Phase 3 / scanner=%s backbone=%s ablation=%s n_runs=%d ===",
             cfg.scanner, args.backbone, name, args.num_runs)

    if args.action in {"train", "both"}:
        for k in range(args.num_runs):
            if not args.retrain_all and _run_already_done(cfg, args.backbone, name, k):
                LOG.info("[%s/%s/run %d] checkpoint already present, skipping",
                         cfg.scanner, name, k)
                continue
            _train_one(cfg, args.backbone, run_idx=k,
                       ablate=args.ablate, ablation_name=name,
                       random_n=args.ablate_random_n,
                       subset_of=args.ablate_subset_of,
                       subset_n=args.subset_n)

    if args.action in {"eval", "both"}:
        per_seed: List[Dict] = []
        for k in range(args.num_runs):
            per_seed.append(_eval_one(cfg, args.backbone, run_idx=k,
                                      ablation_name=name,
                                      eval_batch_size=args.eval_batch_size))
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
