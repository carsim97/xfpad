"""Train g_psi on a REDUCED material vocabulary, as Phase 3b requires.

Phase 4 asks which excluded candidate to add to a reduced training set. The
'xfpad' strategy must therefore read the manifold induced by the REDUCED
vocabulary, not the canonical 7-material one: the recommendation has to be made
with only the information the practitioner would actually have.

Rather than duplicating configs and split files, this filters in memory:
  1. build labels with the FULL config,
  2. keep bona fide + the materials of the reduced vocabulary,
  3. renumber the kept materials to a contiguous 1..K',
  4. train g_psi with K' prototypes on the corresponding cached features.

The checkpoint stores the reduced label->name map, so downstream analysis knows
which prototype is which material without a parallel config.

Usage
-----
    python scripts/train_reduced_encoder.py -c configs/greenbit.yaml \
        --keep Latex RProFast Ecoflex RPro10 --num-runs 10 --device cuda
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import (  # noqa: E402
    base_parser, bona_fide_label, known_names, known_pairs, load_with_overrides,
)
from xfpad.config import Config  # noqa: E402
from xfpad.data import FeatureDataset, build_labels  # noqa: E402
from xfpad.losses import AngularLoss, ConcentricLoss, make_log_scale  # noqa: E402
from xfpad.models import GeometricEncoder  # noqa: E402
from xfpad.utils import (  # noqa: E402
    ensure_dir, features_path, get_logger, read_split, resolve_device, set_seed,
    split_path,
)

LOG = get_logger("reduced_encoder")


def _ckpt(cfg: Config, seed: int) -> Path:
    return Path(cfg.paths.checkpoints) / f"geometric_{cfg.scanner}_reduced_{seed}.pth"


def build_reduced(cfg: Config, keep: list[str]):
    """Return (features, labels_renumbered, reduced_name_map)."""
    feats = np.load(features_path(cfg.paths.features_dir, cfg.scanner, "train"))
    paths = read_split(split_path(cfg.paths.splits_dir, cfg.scanner, "train"))
    labels = np.array(build_labels(paths, known_pairs(cfg)))
    if feats.shape[0] != labels.shape[0]:
        raise ValueError(f"features/labels mismatch: {feats.shape[0]} vs {labels.shape[0]}")

    names = known_names(cfg)
    bf = bona_fide_label(names)
    name2lbl = {v: k for k, v in names.items()}
    unknown = [m for m in keep if m not in name2lbl]
    if unknown:
        raise ValueError(f"unknown material(s) {unknown}; available: {sorted(name2lbl)}")

    keep_lbls = [name2lbl[m] for m in keep]
    mask = np.isin(labels, [bf] + keep_lbls)
    feats_r, labels_r = feats[mask], labels[mask]

    # renumber kept materials to 1..K' preserving the order given in --keep
    remap = {bf: 0}
    reduced_names = {0: names[bf]}
    for new, m in enumerate(keep, start=1):
        remap[name2lbl[m]] = new
        reduced_names[new] = m
    labels_new = np.array([remap[int(l)] for l in labels_r], dtype=np.int64)

    LOG.info("reduced vocabulary: %s  (K=%d)  |  %d/%d samples kept",
             ", ".join(keep), len(keep), mask.sum(), len(labels))
    return feats_r, labels_new, reduced_names


def train_one(cfg: Config, feats, labels, reduced_names, seed: int,
              device_spec: str, epochs: int | None) -> Path:
    out = _ckpt(cfg, seed)
    if out.exists():
        LOG.info("seed %d: checkpoint exists, skipping", seed)
        return out

    device = resolve_device(device_spec)
    set_seed(seed)
    n_epochs = epochs or cfg.geometric.num_epochs
    K = int(labels.max())

    loader = DataLoader(
        FeatureDataset(feats, torch.tensor(labels, dtype=torch.long)),
        batch_size=cfg.geometric.batch_size, shuffle=True,
        num_workers=cfg.geometric.num_workers)

    model = GeometricEncoder(dropout=cfg.geometric.dropout).to(device)
    angular = AngularLoss(make_log_scale(gamma_min=cfg.loss.gamma_min,
                                         gamma_max=cfg.loss.gamma_max)).to(device)
    angular.update_K(K, device)
    concentric = ConcentricLoss(rho_bf=cfg.loss.rho_bf,
                                delta_rho=cfg.loss.delta_rho).to(device)
    opt = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": angular.parameters()}],
        lr=cfg.geometric.lr, weight_decay=cfg.geometric.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    for epoch in range(n_epochs):
        model.train()
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z = model(x)
            loss = angular(z, y) + concentric(z, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss)
        sched.step()
        if (epoch + 1) % 40 == 0 or epoch == 0:
            LOG.info("seed %d epoch %d/%d loss=%.4f", seed, epoch + 1, n_epochs, running)

    ensure_dir(out.parent)
    torch.save({"model": model.state_dict(), "angular": angular.state_dict(),
                "K": K, "seed": seed,
                "reduced_names": reduced_names}, out)
    LOG.info("saved -> %s", out.name)
    return out


def main() -> None:
    p = base_parser("Train g_psi on a reduced material vocabulary.")
    p.add_argument("--keep", nargs="+", required=True,
                   help="Materials to KEEP, by name (e.g. --keep Latex RProFast "
                        "Ecoflex RPro10). Bona fide is always kept.")
    p.add_argument("--num-runs", type=int, default=10)
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()

    cfg = load_with_overrides(args)
    feats, labels, names = build_reduced(cfg, args.keep)
    for k in range(args.num_runs):
        train_one(cfg, feats, labels, names, cfg.seed + k,
                  args.device or cfg.device, args.epochs)


if __name__ == "__main__":
    main()
