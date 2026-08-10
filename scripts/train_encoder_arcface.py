"""Diagnostic baseline: ArcFace/CosFace free-prototype encoder.

Replaces X-FPAD's deterministic *uniform* angular allocation
(phi_k = 2*pi*k/K + theta_offset, a single shared learnable scalar) with
K FREE, independently learnable 2-D prototypes, supervised by a CosFace
additive-cosine-margin softmax (ArcFace is a one-line change: apply the
margin on the angle instead of the cosine). This is the "prototype-based
metric-learning alternative" to the uniform allocation: prototypes are
scattered by optimisation rather than placed on a fixed uniform grid, so
the manifold loses the guaranteed 2*pi/K separation and the reproducible
angular hierarchy (paper's structural properties (4)-(5), Sec. III-E).

The radial term (ConcentricLoss) is kept identical to X-FPAD, so this
ablation isolates ONLY the effect of the angular-allocation strategy on
the directional attribution p_{u,k}.

The checkpoint is drop-in compatible with
`scripts/baselines_attribution.py --reductions xfpad --geo-ckpt <path>`
(the attribution uses empirical centroid directions, exactly as for X-FPAD).

Usage
-----
    CUDA_VISIBLE_DEVICES="" python scripts/train_encoder_arcface.py \
        -c configs/greenbit.yaml --device cpu --num-runs 1 --scale 10 --margin 0.2

    python scripts/baselines_attribution.py -c configs/greenbit.yaml \
        --reductions xfpad \
        --geo-ckpt checkpoints/geometric_greenbit_arcface_0.pth
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

LOG = get_logger("arcface")


# ---------------------------------------------------------------------------
# CosFace / ArcFace angular loss with FREE learnable prototypes
# ---------------------------------------------------------------------------

class FreePrototypeMarginLoss(nn.Module):
    """Margin-softmax over K free learnable 2-D prototypes.

    variant='cosface' : additive cosine margin  phi = cos(theta) - m   (default)
    variant='arcface' : additive angular margin phi = cos(theta + m)
    Applied to attack samples (y != 0) only, mirroring X-FPAD's L_cos.
    """

    def __init__(self, K: int, scale: float = 10.0, margin: float = 0.2,
                 variant: str = "cosface") -> None:
        super().__init__()
        self.K = int(K)
        self.scale = float(scale)
        self.margin = float(margin)
        self.variant = variant
        # Free prototypes: no uniform-spacing constraint, random init.
        self.prototypes = nn.Parameter(torch.randn(self.K, 2) * 0.1)

    def normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=1)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = labels != 0
        if mask.sum() == 0:
            return torch.zeros((), device=z.device)

        z_sel = F.normalize(z[mask], dim=1)         # (n, 2) on unit circle
        y_sel = labels[mask] - 1                    # 0..K-1
        P = self.normalized_prototypes()            # (K, 2)

        cos = z_sel @ P.t()                         # (n, K) cosine similarities
        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if self.variant == "arcface":
            theta = torch.acos(cos)
            target_phi = torch.cos(theta + self.margin)
        else:  # cosface
            target_phi = cos - self.margin

        one_hot = F.one_hot(y_sel, num_classes=self.K).float()
        logits = self.scale * (one_hot * target_phi + (1.0 - one_hot) * cos)
        return F.cross_entropy(logits, y_sel)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _ckpt_path(cfg: Config, seed: int, variant: str) -> Path:
    tag = "arcface" if variant == "arcface" else "cosface"
    return Path(cfg.paths.checkpoints) / f"geometric_{cfg.scanner}_{tag}_{seed}.pth"


def train_one(cfg: Config, seed: int, device_spec: str,
              scale: float, margin: float, variant: str,
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

    LOG.info("%s encoder seed=%d device=%s: %d samples, K=%d (free prototypes, s=%.1f m=%.2f)",
             variant.upper(), seed, device, len(labels), K, scale, margin)

    loader = DataLoader(
        FeatureDataset(features, torch.tensor(labels, dtype=torch.long)),
        batch_size=cfg.geometric.batch_size,
        shuffle=True,
        num_workers=cfg.geometric.num_workers,
    )

    model = GeometricEncoder(dropout=cfg.geometric.dropout).to(device)
    angular = FreePrototypeMarginLoss(K, scale=scale, margin=margin,
                                      variant=variant).to(device)
    concentric = ConcentricLoss(
        rho_bf=cfg.loss.rho_bf,
        delta_rho=cfg.loss.delta_rho,
    ).to(device)

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": angular.parameters()}],
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
            loss = angular(z, y) + concentric(z, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            LOG.info("%s seed=%d epoch %d/%d  loss=%.4f",
                     variant, seed, epoch + 1, num_epochs, running)

    # Report the learned prototype angles to show they are NOT uniform.
    with torch.no_grad():
        P = angular.normalized_prototypes().cpu().numpy()
        angles = np.degrees(np.arctan2(P[:, 1], P[:, 0])) % 360.0
        LOG.info("Learned prototype angles (deg): %s",
                 np.array2string(np.sort(angles), precision=1))

    out = _ckpt_path(cfg, seed, variant)
    ensure_dir(out.parent)
    torch.save({
        "model": model.state_dict(),
        "prototypes": angular.prototypes.detach().cpu(),
        "threshold": concentric.threshold,
        "K": K,
        "seed": seed,
        "variant": variant,
        "scale": scale,
        "margin": margin,
    }, out)
    LOG.info("Saved %s encoder -> %s", variant, out)
    return out


def main() -> None:
    parser = base_parser("Free-prototype ArcFace/CosFace encoder (diagnostic baseline).")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Explicit seed list (overrides --num-runs); e.g. "
                             "--seeds 9 to resume a single missing seed.")
    parser.add_argument("--device", default=None,
                        help="Override cfg.device ('cpu'|'cuda'|'auto').")
    parser.add_argument("--variant", choices=["cosface", "arcface"],
                        default="cosface", help="Margin type (default cosface).")
    parser.add_argument("--scale", type=float, default=10.0,
                        help="Softmax scale s (default 10).")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="Additive margin m (default 0.2).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override cfg.geometric.num_epochs (e.g. for smoke tests).")
    args = parser.parse_args()

    cfg = load_with_overrides(args)
    device_spec = args.device or cfg.device
    seeds = args.seeds if args.seeds is not None else [cfg.seed + k for k in range(args.num_runs)]
    LOG.info("=== %s encoder / scanner=%s seeds=%s ===", args.variant.upper(), cfg.scanner, seeds)
    for s in seeds:
        train_one(cfg, seed=s, device_spec=device_spec,
                  scale=args.scale, margin=args.margin, variant=args.variant,
                  epochs=args.epochs)


if __name__ == "__main__":
    main()
