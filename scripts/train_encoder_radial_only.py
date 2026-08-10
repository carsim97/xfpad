"""Diagnostic baseline: radial-only geometric encoder.

Trains g_psi with the Concentric radial loss ONLY — the Angular loss
L_cos is removed. Bona fide samples are still compacted at the origin and
PAIs pushed beyond rho_pa, but the PAI classes receive NO angular
supervision, so they are free to occupy arbitrary directions. The purpose
is to show that, without the angular term, the directional attribution
p_{u,k} degenerates (the empirical PAI centroids no longer span dedicated
angular sectors), isolating the contribution of L_cos to the manifold's
diagnostic value.

The produced checkpoint is drop-in compatible with
`scripts/baselines_attribution.py --reductions xfpad --geo-ckpt <path>`
(only the "model" key is required by the attribution comparison).

Usage
-----
    # GPU when free, or force CPU to avoid contending with training jobs:
    CUDA_VISIBLE_DEVICES="" python scripts/train_encoder_radial_only.py \
        -c configs/greenbit.yaml --device cpu --num-runs 1

    # then feed it to the baseline comparison:
    python scripts/baselines_attribution.py -c configs/greenbit.yaml \
        --reductions xfpad \
        --geo-ckpt checkpoints/geometric_greenbit_radialonly_0.pth
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import (  # noqa: E402
    base_parser,
    known_pairs,
    load_with_overrides,
)
from xfpad.config import Config  # noqa: E402
from xfpad.data import (  # noqa: E402
    FeatureDataset,
    assert_contiguous_pai_labels,
    build_labels,
)
from xfpad.losses import ConcentricLoss  # noqa: E402
from xfpad.models import GeometricEncoder  # noqa: E402
from xfpad.utils import (  # noqa: E402
    ensure_dir,
    features_path,
    get_logger,
    read_split,
    resolve_device,
    set_seed,
    split_path,
)

LOG = get_logger("radial_only")


def _ckpt_path(cfg: Config, seed: int) -> Path:
    return Path(cfg.paths.checkpoints) / f"geometric_{cfg.scanner}_radialonly_{seed}.pth"


def train_one(cfg: Config, seed: int, device_spec: str,
              epochs: int | None = None) -> Path:
    device = resolve_device(device_spec)
    set_seed(seed)
    num_epochs = epochs or cfg.geometric.num_epochs

    fpath = features_path(cfg.paths.features_dir, cfg.scanner, "train")
    if not fpath.exists():
        raise FileNotFoundError(f"Train features not found: {fpath}. "
                                "Run phase1 --stage features first.")
    features = np.load(fpath)

    paths = read_split(split_path(cfg.paths.splits_dir, cfg.scanner, "train"))
    labels = build_labels(paths, known_pairs(cfg))
    K = assert_contiguous_pai_labels(labels)
    if features.shape[0] != len(labels):
        raise ValueError(f"Feature/label mismatch: {features.shape[0]} vs {len(labels)}.")

    LOG.info("RADIAL-ONLY encoder seed=%d device=%s: %d samples, K=%d (no angular loss)",
             seed, device, len(labels), K)

    loader = DataLoader(
        FeatureDataset(features, torch.tensor(labels, dtype=torch.long)),
        batch_size=cfg.geometric.batch_size,
        shuffle=True,
        num_workers=cfg.geometric.num_workers,
    )

    model = GeometricEncoder(dropout=cfg.geometric.dropout).to(device)
    concentric = ConcentricLoss(
        rho_bf=cfg.loss.rho_bf,
        delta_rho=cfg.loss.delta_rho,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.geometric.lr,
        weight_decay=cfg.geometric.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs,
    )

    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            z = model(x)
            loss = concentric(z, y)            # <-- radial term ONLY
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            LOG.info("radial-only seed=%d epoch %d/%d  loss=%.4f",
                     seed, epoch + 1, num_epochs, running)

    out = _ckpt_path(cfg, seed)
    ensure_dir(out.parent)
    torch.save({
        "model": model.state_dict(),
        "angular": None,                       # no angular geometry
        "threshold": concentric.threshold,
        "K": K,
        "seed": seed,
        "variant": "radial_only",
    }, out)
    LOG.info("Saved radial-only encoder -> %s", out)
    return out


def main() -> None:
    parser = base_parser("Radial-only geometric encoder (diagnostic baseline).")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of seeds to train (cfg.seed, cfg.seed+1, ...).")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Explicit seed list (overrides --num-runs); e.g. "
                             "--seeds 9 to resume a single missing seed.")
    parser.add_argument("--device", default=None,
                        help="Override cfg.device ('cpu'|'cuda'|'auto').")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override cfg.geometric.num_epochs (e.g. for smoke tests).")
    args = parser.parse_args()

    cfg = load_with_overrides(args)
    device_spec = args.device or cfg.device
    seeds = args.seeds if args.seeds is not None else [cfg.seed + k for k in range(args.num_runs)]
    LOG.info("=== Radial-only encoder / scanner=%s seeds=%s ===", cfg.scanner, seeds)
    for s in seeds:
        train_one(cfg, seed=s, device_spec=device_spec, epochs=args.epochs)


if __name__ == "__main__":
    main()
