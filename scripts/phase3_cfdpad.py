"""Phase 3 for CFD-PAD, the fourth audited system.

A separate entry point from phase3_audit_pad.py, whose --backbone dispatch
builds single-loss classifiers: CFD-PAD has a three-term objective and a
channel-importance pass that runs inside the training loop. Everything that
could bias the comparison is shared -- splits, labels, RAM-cached dataset,
APCER evaluation and JSON layout -- so its cells feed correlate_attr_apcer.py
exactly like those of the generic backbones.

Usage
-----
    # cost measurement (few epochs, one seed)
    python scripts/phase3_cfdpad.py -c configs/greenbit_cons.yaml \
        --action train --num-runs 1 --epochs 2 --importance-every 1 --measure

    # production cell
    python scripts/phase3_cfdpad.py -c configs/greenbit_cons.yaml \
        --action both --num-runs 10 --ablation-name baseline \
        --save-json outputs/lomo/phase3_greenbit_cfd_pad_baseline.json
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import base_parser, known_pairs, load_with_overrides  # noqa: E402
from scripts.phase3_audit_pad import (  # noqa: E402
    _aggregate_eval,
    _apply_ablation,
    _apply_random_ablation,
    _binary_label_for_path,
    _per_pai_definitions,
)
from xfpad.config import Config  # noqa: E402
from xfpad.data import FingerprintDataset, build_labels  # noqa: E402
from xfpad.metrics import apcer_bpcer, apcer_per_unseen_pai  # noqa: E402
from xfpad.models.cfd_pad import CFDPad, PAAdaptationLoss  # noqa: E402
from xfpad.utils import (  # noqa: E402
    ensure_dir, get_logger, read_split, resolve_device, set_seed, split_path,
)

LOG = get_logger("cfdpad")


def _ckpt(cfg: Config, ablation: str, run: int) -> Path:
    return (Path(cfg.paths.checkpoints) / "pad" / cfg.scanner / ablation /
            f"cfd_pad_run{run}.pth")


def _run_already_done(cfg: Config, ablation: str, run_idx: int) -> bool:
    """True if this seed's checkpoint exists AND is loadable.

    Same contract as phase3_audit_pad._run_already_done, and it matters more
    here: a CFD-PAD seed is the most expensive of the four systems, so without
    per-seed resume an interruption would restart the cell from seed 0. A
    checkpoint truncated mid-write fails to load and is simply retrained.
    """
    path = _ckpt(cfg, ablation, run_idx)
    if not path.exists():
        return False
    try:
        torch.load(path, map_location="cpu")
        return True
    except Exception as e:                                  # noqa: BLE001
        LOG.warning("Checkpoint %s unreadable (%s); it will be retrained.",
                    path.name, e)
        return False


def train_one(cfg: Config, run_idx: int, ablate: List[str], ablation: str,
              epochs: int | None, imp_every: int, imp_chunk: int,
              k: int, w_triplet: float, w_masked: float,
              measure: bool, random_n: int | None = None) -> Path:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed + run_idx)
    n_epochs = epochs or cfg.pad_detector.num_epochs

    raw = read_split(split_path(cfg.paths.splits_dir, cfg.scanner, "train"))
    if random_n:
        # Matched-size control. The draw must be the SAME one the generic
        # backbones saw, or this would not be the same ablation: same starting
        # list, same rng seeded with cfg.seed + run_idx, as in
        # phase3_audit_pad.py.
        paths = _apply_random_ablation(raw, random_n,
                                       random.Random(cfg.seed + run_idx), cfg)
    else:
        paths = _apply_ablation(raw, ablate)
    # The dataset carries the MATERIAL id (0 = bona fide, 1..K = PAI); the binary
    # target is derived from it, so both labels survive shuffling. Cross-checked
    # against the path-based binary rule below.
    y_mat = build_labels(paths, known_pairs(cfg))
    y_bin_ref = [_binary_label_for_path(p) for p in paths]
    derived = [1 if m == 0 else 0 for m in y_mat]
    if derived != y_bin_ref:
        n_bad = sum(a != b for a, b in zip(derived, y_bin_ref))
        raise ValueError(f"material/binary label mismatch on {n_bad} paths; "
                         "check the config mapping before training.")

    ds = FingerprintDataset(paths, y_mat, cache_in_ram=True)
    loader = DataLoader(ds, batch_size=cfg.pad_detector.batch_size, shuffle=True,
                        num_workers=cfg.pad_detector.num_workers, pin_memory=True)

    model = CFDPad(in_channels=cfg.backbone.in_channels, k=k).to(device)
    bce = nn.BCEWithLogitsLoss()
    padp = PAAdaptationLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.pad_detector.lr,
                           weight_decay=cfg.pad_detector.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    LOG.info("[%s/%s/run %d] CFD-PAD: %d samples, k=%d, importance every %d batch(es)",
             cfg.scanner, ablation, run_idx, len(paths), k, imp_every)

    epoch_times = []
    for epoch in range(n_epochs):
        model.train()
        model.reset_channel_gap()          # gap is accumulated per epoch
        t0, running, nb = time.time(), 0.0, 0
        for bi, (x, ym) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            ym = ym.to(device)                       # material id
            yb = (ym == 0).float().unsqueeze(1)      # 1 = bona fide

            f = model.front(x)
            e = model.embed(f)
            o = model.classify(e)

            if bi % imp_every == 0:
                model.update_channel_gap(f.detach(), o.detach(), chunk=imp_chunk)

            loss = bce(o, yb)
            if model.channel_gap.sum() > 0:
                e2 = model.embed(model.denoise(f))
                loss = (loss
                        + w_triplet * padp(e2, ym)
                        + w_masked * bce(model.classify(e2), yb))

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss)
            nb += 1
        sched.step()
        dt = time.time() - t0
        epoch_times.append(dt)
        LOG.info("[run %d] epoch %d/%d loss=%.4f  %.1fs (%.3fs/batch)",
                 run_idx, epoch + 1, n_epochs, running, dt, dt / max(nb, 1))

    if measure:
        keep = torch.topk(model.channel_gap, k).indices.sort().values.tolist()
        print(f"\n[MEASURE] importance_every={imp_every}  "
              f"mean epoch = {np.mean(epoch_times):.1f}s  "
              f"(batches={nb}, batch_size={loader.batch_size})")
        print(f"[MEASURE] top-{k} channels: {keep}")

    out = _ckpt(cfg, ablation, run_idx)
    ensure_dir(out.parent)
    # Written via a temporary file, so the checkpoint is either absent or
    # complete: an interruption during torch.save would otherwise leave a
    # truncated .pth for _run_already_done to detect and retrain.
    tmp = out.with_suffix(".pth.tmp")
    torch.save({"model": model.state_dict(), "backbone": "cfd_pad",
                "ablation": ablation, "run": run_idx, "k": k,
                "importance_every": imp_every}, tmp)
    tmp.replace(out)
    return out


def eval_one(cfg: Config, run_idx: int, ablation: str, k: int,
             batch_size: int = 64) -> Dict:
    device = resolve_device(cfg.device)
    ck = _ckpt(cfg, ablation, run_idx)
    model = CFDPad(in_channels=cfg.backbone.in_channels, k=k).to(device)
    model.load_state_dict(torch.load(ck, map_location=device)["model"])
    model.eval()

    paths = read_split(split_path(cfg.paths.splits_dir, cfg.scanner, "test"))
    labels = [_binary_label_for_path(p) for p in paths]   # eval stays binary
    loader = DataLoader(FingerprintDataset(paths, labels, cache_in_ram=True),
                        batch_size=batch_size, shuffle=False,
                        num_workers=cfg.pad_detector.num_workers, pin_memory=True)
    preds: List[int] = []
    with torch.no_grad():
        for x, _ in loader:
            p = torch.sigmoid(model(x.to(device))).squeeze(1)
            preds.extend((p > cfg.pad_detector.threshold).long().cpu().tolist())

    apcer, bpcer, ace = apcer_bpcer(np.array(labels), np.array(preds))
    per_pai = apcer_per_unseen_pai(paths, preds, _per_pai_definitions(cfg))
    LOG.info("[%s/%s/run %d] APCER=%.2f BPCER=%.2f", cfg.scanner, ablation, run_idx, apcer, bpcer)
    return {"run": run_idx, "backbone": "cfd_pad", "ablation": ablation,
            "APCER": apcer, "BPCER": bpcer, "ACE": ace, "per_pai_apcer": per_pai}


def main() -> None:
    p = base_parser("Phase 3 for CFD-PAD.")
    p.add_argument("--action", required=True, choices=["train", "eval", "both"])
    p.add_argument("--num-runs", type=int, default=10)
    p.add_argument("--epochs", type=int, default=None, help="override epochs")
    p.add_argument("--ablate", nargs="*", default=[])
    p.add_argument("--ablate-random-n", type=int, default=None,
                   help="Matched-size random control: remove N spoof samples "
                        "chosen uniformly at random (independent draw per "
                        "seed, identical to the generic backbones).")
    p.add_argument("--ablation-name", default="baseline")
    p.add_argument("--k", type=int, default=30, help="important channels (paper: 30/160)")
    p.add_argument("--importance-every", type=int, default=1,
                   help="run the channel-importance pass every N batches "
                        "(1 = faithful to the paper; higher = cheaper)")
    p.add_argument("--importance-chunk", type=int, default=16,
                   help="channels processed per chunk (memory bound)")
    p.add_argument("--w-triplet", type=float, default=1.0)
    p.add_argument("--w-masked", type=float, default=1.0)
    p.add_argument("--measure", action="store_true",
                   help="print epoch timings and the selected channels")
    p.add_argument("--retrain-all", action="store_true",
                   help="retrain every seed even if its checkpoint exists. By "
                        "default a seed whose checkpoint is present and loadable "
                        "is skipped, so an interrupted cell resumes where it "
                        "stopped instead of restarting from seed 0.")
    p.add_argument("--save-json", default=None)
    args = p.parse_args()
    if args.ablate and args.ablate_random_n:
        p.error("--ablate e --ablate-random-n si escludono a vicenda")
    cfg = load_with_overrides(args)

    if args.action in {"train", "both"}:
        for r in range(args.num_runs):
            if not args.retrain_all and _run_already_done(cfg, args.ablation_name, r):
                LOG.info("[%s/%s/run %d] checkpoint already present, skipping",
                         cfg.scanner, args.ablation_name, r)
                continue
            train_one(cfg, r, args.ablate, args.ablation_name, args.epochs,
                      args.importance_every, args.importance_chunk, args.k,
                      args.w_triplet, args.w_masked, args.measure,
                      random_n=args.ablate_random_n)

    if args.action in {"eval", "both"}:
        res = [eval_one(cfg, r, args.ablation_name, args.k) for r in range(args.num_runs)]
        aggr = _aggregate_eval(res)
        for name, v in aggr["per_pai"].items():
            print(f"  {name:<30} {v['mean']:.2f}+/-{v['std']:.2f}")
        if args.save_json:
            out = Path(args.save_json)
            ensure_dir(out.parent)
            # Atomic, for the same reason as the checkpoint: the cell JSON is
            # what makes the driver skip a finished cell, so a truncated one
            # would both fail to parse and hide the work already done.
            tmp = out.with_suffix(".json.tmp")
            with tmp.open("w") as fh:
                json.dump({"per_seed": res, "aggregated": aggr}, fh, indent=2)
            tmp.replace(out)
            LOG.info("saved -> %s", out)


if __name__ == "__main__":
    main()
