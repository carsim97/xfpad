"""Phase 1 — Manifold construction.

Three sequential stages, runnable independently or all together via --stage all:

  1. backbone : train MobileNet-v2 multi-class feature extractor f_phi
                on minutiae-guided patches. Saves
                checkpoints/feature_extractor_<scanner>.pth.

  2. features : run the frozen f_phi on the train and (optionally) test
                splits, saving 1280-D embeddings under data/features/.

  3. encoder  : train the geometric encoder g_psi on the cached embeddings
                with the compound loss L_conc + L_cos. By default trains
                a single seed (cfg.seed); use --num-runs for multi-seed.

Usage examples
--------------
    python scripts/phase1_train.py -c configs/greenbit.yaml --stage all
    python scripts/phase1_train.py -c configs/dermalog.yaml --stage encoder \\
                                   --num-runs 10 --resume
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import (  # noqa: E402
    base_parser,
    known_names,
    known_pairs,
    load_with_overrides,
)
from xfpad.config import Config  # noqa: E402
from xfpad.data import (  # noqa: E402
    FeatureDataset,
    FingerprintDataset,
    assert_contiguous_pai_labels,
    build_labels,
)
from xfpad.losses import AngularLoss, ConcentricLoss, make_log_scale  # noqa: E402
from xfpad.models import FeatureExtractor, GeometricEncoder  # noqa: E402
from xfpad.utils import (  # noqa: E402
    ensure_dir,
    feature_extractor_ckpt,
    features_path,
    geometric_ckpt,
    get_logger,
    read_split,
    resolve_device,
    set_seed,
    split_path,
)

LOG = get_logger("phase1")


# ===========================================================================
# Stage 1 — Backbone training
# ===========================================================================

def train_backbone(cfg: Config) -> Path:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    pairs = known_pairs(cfg)
    num_classes = len(set(pairs.values()))

    paths_train = read_split(split_path(cfg.paths.splits_dir, cfg.scanner, "train"))
    labels_train = build_labels(paths_train, pairs)
    assert_contiguous_pai_labels(labels_train)

    LOG.info("Backbone training: %d samples, %d classes", len(paths_train), num_classes)

    dataset = FingerprintDataset(paths_train, labels_train)
    loader = DataLoader(
        dataset,
        batch_size=cfg.backbone.batch_size,
        shuffle=True,
        num_workers=cfg.backbone.num_workers,
        pin_memory=True,
    )

    model = FeatureExtractor(
        num_classes=num_classes,
        in_channels=cfg.backbone.in_channels,
        training_mode=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.backbone.lr,
        weight_decay=cfg.backbone.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.backbone.num_epochs,
    )

    for epoch in range(cfg.backbone.num_epochs):
        model.train()
        running = 0.0
        for x, y in tqdm(loader, desc=f"epoch {epoch + 1}"):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        LOG.info("backbone epoch %d/%d  loss=%.4f",
                 epoch + 1, cfg.backbone.num_epochs, running)

    out = feature_extractor_ckpt(cfg.paths.checkpoints, cfg.scanner)
    ensure_dir(out.parent)
    torch.save({"model": model.state_dict()}, out)
    LOG.info("Saved backbone checkpoint -> %s", out)
    return out


# ===========================================================================
# Stage 2 — Feature extraction
# ===========================================================================

def extract_features(cfg: Config, modes: list[str]) -> list[Path]:
    """Run frozen f_phi on the requested splits and save .npy embeddings."""
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    ckpt_path = feature_extractor_ckpt(cfg.paths.checkpoints, cfg.scanner)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Backbone checkpoint not found: {ckpt_path}. Run --stage backbone first."
        )

    model = FeatureExtractor(
        in_channels=cfg.backbone.in_channels,
        training_mode=False,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    written: list[Path] = []
    for mode in modes:
        spath = split_path(cfg.paths.splits_dir, cfg.scanner, mode)
        if not spath.exists():
            LOG.warning("Skipping mode '%s' (split file missing: %s)", mode, spath)
            continue
        paths = read_split(spath)

        # Dummy labels — we just need ordered iteration.
        dataset = FingerprintDataset(paths, [0] * len(paths))
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.backbone.num_workers,
            pin_memory=True,
        )

        feats: list[np.ndarray] = []
        with torch.no_grad():
            for x, _ in tqdm(loader, desc=f"extracting {mode}"):
                emb = model(x.to(device))
                feats.append(emb.cpu().numpy())

        arr = np.concatenate(feats, axis=0)
        out = features_path(cfg.paths.features_dir, cfg.scanner, mode)
        ensure_dir(out.parent)
        np.save(out, arr)
        LOG.info("Saved features %s -> %s (shape %s)", mode, out, arr.shape)
        written.append(out)

    return written


# ===========================================================================
# Stage 3 — Geometric encoder training
# ===========================================================================

def train_geometric_encoder(cfg: Config,
                            seed: int,
                            resume: bool = False) -> Path:
    device = resolve_device(cfg.device)
    set_seed(seed)

    fpath = features_path(cfg.paths.features_dir, cfg.scanner, "train")
    if not fpath.exists():
        raise FileNotFoundError(
            f"Train features not found: {fpath}. Run --stage features first."
        )
    features = np.load(fpath)

    paths = read_split(split_path(cfg.paths.splits_dir, cfg.scanner, "train"))
    pairs = known_pairs(cfg)
    labels = build_labels(paths, pairs)
    K = assert_contiguous_pai_labels(labels)
    LOG.info("Encoder seed=%d: %d samples, K=%d PAI prototypes",
             seed, len(labels), K)

    if features.shape[0] != len(labels):
        raise ValueError(
            f"Feature/label length mismatch: {features.shape[0]} vs {len(labels)}. "
            "Re-extract features after changing splits or label mapping."
        )

    labels_t = torch.tensor(labels, dtype=torch.long)
    loader = DataLoader(
        FeatureDataset(features, labels_t),
        batch_size=cfg.geometric.batch_size,
        shuffle=True,
        num_workers=cfg.geometric.num_workers,
    )

    model = GeometricEncoder(dropout=cfg.geometric.dropout).to(device)
    log_scale = make_log_scale(
        gamma_min=cfg.loss.gamma_min,
        gamma_max=cfg.loss.gamma_max,
    )
    angular = AngularLoss(log_scale).to(device)
    angular.update_K(K, device)
    concentric = ConcentricLoss(
        rho_bf=cfg.loss.rho_bf,
        delta_rho=cfg.loss.delta_rho,
    ).to(device)

    ckpt_path = geometric_ckpt(cfg.paths.checkpoints, cfg.scanner, seed)
    if resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        if state.get("K") == K:
            angular.load_state_dict(state["angular"])
        else:
            LOG.info("K changed (%s -> %d): resetting angular geometry.",
                     state.get("K"), K)
            angular.update_K(K, device)

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": angular.parameters()}],
        lr=cfg.geometric.lr,
        weight_decay=cfg.geometric.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.geometric.num_epochs,
    )

    for epoch in range(cfg.geometric.num_epochs):
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
            LOG.info("encoder seed=%d epoch %d/%d  loss=%.4f  theta_offset=%.4f",
                     seed, epoch + 1, cfg.geometric.num_epochs,
                     running, angular.theta_offset.item())

    ensure_dir(ckpt_path.parent)
    torch.save({
        "model": model.state_dict(),
        "angular": angular.state_dict(),
        "threshold": concentric.threshold,
        "K": K,
        "seed": seed,
    }, ckpt_path)
    LOG.info("Saved encoder checkpoint -> %s", ckpt_path)
    return ckpt_path


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = base_parser("Phase 1 - Manifold construction.")
    parser.add_argument(
        "--stage", required=True,
        choices=["backbone", "features", "encoder", "all"],
        help="Which sub-stage to run (or 'all' for the full pipeline).",
    )
    parser.add_argument(
        "--num-runs", type=int, default=1,
        help="Encoder only: number of seeds to train (cfg.seed, cfg.seed+1, ...). "
             "Default 1.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Encoder only: resume from existing checkpoint if present.",
    )
    parser.add_argument(
        "--features-modes", nargs="+", default=["train", "test"],
        help="Features only: which splits to extract (default: both).",
    )

    args = parser.parse_args()
    cfg = load_with_overrides(args)

    LOG.info("=== Phase 1 / scanner=%s ===", cfg.scanner)

    if args.stage in {"backbone", "all"}:
        train_backbone(cfg)

    if args.stage in {"features", "all"}:
        extract_features(cfg, args.features_modes)

    if args.stage in {"encoder", "all"}:
        for k in range(args.num_runs):
            seed = cfg.seed + k
            train_geometric_encoder(cfg, seed=seed, resume=args.resume)


if __name__ == "__main__":
    main()
