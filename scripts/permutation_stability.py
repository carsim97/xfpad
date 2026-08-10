"""Semantic stability of the manifold under prototype permutation.

Materials are assigned to angular sectors by label index, so the attribution
could in principle follow that arbitrary ordering rather than the geometry.
This script retrains the geometric encoder g_psi under several permutations of the
material -> angular-sector assignment and checks whether the directional
attribution p_{u,k} (by material NAME) is preserved.

Key design point
----------------
The permutation is applied ONLY to the labels seen by the angular loss during
training (it changes which sector each material is pushed toward). The
attribution p_{u,k} is then computed with the ORIGINAL material labels via
empirical centroid directions, so it is expressed per real material. If the
manifold's diagnostic value is intrinsic (not an artefact of the angular
layout), the ranked anchors of each unseen PAI stay stable across permutations.

Permutations compared (reference = identity):
  * identity        : the canonical assignment used in the paper;
  * cosine_optimal  : the ordering from optimize_ordering.py (if available);
  * random_XX       : uniformly random permutations (seeded, reproducible).

Metrics vs the identity reference, per unseen PAI then averaged:
  * top1_match      : same #1 anchor;
  * top2_match      : same (unordered) top-2 anchor set;
  * kendall_tau     : rank correlation of the full p_{u,.} weight vector.

CPU-only by design (reads cached features). Idempotent: retrained encoders are
cached as checkpoints/geometric_<scanner>_perm<name>_<seed>.pth and skipped if
present; results go to outputs/point_d/.

Usage
-----
    CUDA_VISIBLE_DEVICES="" python scripts/permutation_stability.py \
        -c configs/greenbit.yaml --device cpu --num-permutations 8 \
        --save-json outputs/point_d/permutation_greenbit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

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
from xfpad.config import Config  # noqa: E402
from xfpad.data import (  # noqa: E402
    FeatureDataset,
    assert_contiguous_pai_labels,
    build_labels,
)
from xfpad.losses import AngularLoss, ConcentricLoss, make_log_scale  # noqa: E402
from xfpad.metrics import analyze_unseen_pais  # noqa: E402
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

LOG = get_logger("perm_stability")


# ---------------------------------------------------------------------------
# Permutation-aware g_psi training
# ---------------------------------------------------------------------------

def _ckpt_path(cfg: Config, perm_name: str, seed: int) -> Path:
    return Path(cfg.paths.checkpoints) / f"geometric_{cfg.scanner}_perm{perm_name}_{seed}.pth"


def train_permuted_encoder(cfg: Config, features: np.ndarray, labels: np.ndarray,
                           perm: np.ndarray, perm_name: str, seed: int,
                           device_spec: str, epochs: int | None) -> Path:
    """Train g_psi where PAI label l is pushed toward sector perm[l-1].

    perm : length-K array, a permutation of 1..K. Bona fide (0) is untouched.
    Returns the checkpoint path (cached / skipped if it already exists).
    """
    out = _ckpt_path(cfg, perm_name, seed)
    if out.exists():
        LOG.info("[%s/perm=%s/seed=%d] checkpoint exists, skip", cfg.scanner, perm_name, seed)
        return out

    device = resolve_device(device_spec)
    set_seed(seed)
    num_epochs = epochs or cfg.geometric.num_epochs
    K = int(labels.max())

    # Relabel PAI samples for the ANGULAR loss only (0 stays 0).
    ang_labels = labels.copy()
    pai_mask = labels != 0
    ang_labels[pai_mask] = perm[labels[pai_mask] - 1]

    loader = DataLoader(
        FeatureDataset(features, torch.tensor(ang_labels, dtype=torch.long)),
        batch_size=cfg.geometric.batch_size, shuffle=True,
        num_workers=cfg.geometric.num_workers,
    )

    model = GeometricEncoder(dropout=cfg.geometric.dropout).to(device)
    angular = AngularLoss(make_log_scale(gamma_min=cfg.loss.gamma_min,
                                         gamma_max=cfg.loss.gamma_max)).to(device)
    angular.update_K(K, device)
    concentric = ConcentricLoss(rho_bf=cfg.loss.rho_bf,
                                delta_rho=cfg.loss.delta_rho).to(device)
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": angular.parameters()}],
        lr=cfg.geometric.lr, weight_decay=cfg.geometric.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    LOG.info("[%s/perm=%s/seed=%d] training g_psi (K=%d, %d epochs) map=%s",
             cfg.scanner, perm_name, seed, K, num_epochs, perm.tolist())
    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z = model(x)
            loss = angular(z, y) + concentric(z, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        if (epoch + 1) % 40 == 0 or epoch == 0:
            LOG.info("[%s/perm=%s/seed=%d] epoch %d/%d loss=%.4f",
                     cfg.scanner, perm_name, seed, epoch + 1, num_epochs, running)

    ensure_dir(out.parent)
    torch.save({"model": model.state_dict(), "angular": angular.state_dict(),
                "K": K, "seed": seed, "perm": perm.tolist(), "perm_name": perm_name}, out)
    LOG.info("[%s/perm=%s/seed=%d] saved -> %s", cfg.scanner, perm_name, seed, out)
    return out


# ---------------------------------------------------------------------------
# Attribution for a trained (possibly permuted) encoder
# ---------------------------------------------------------------------------

def _project(ckpt: Path, features: np.ndarray, dropout: float) -> np.ndarray:
    model = GeometricEncoder(dropout=dropout)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(features).float()).cpu().numpy()


def attribution_for(ckpt: Path, cfg: Config, feats_train, y_train, feats_test, y_test,
                    train_names_d, test_names_d) -> Dict[str, List]:
    """Return {unseen_name: ranked [(material, weight), ...]} by real material."""
    z_train = _project(ckpt, feats_train, cfg.geometric.dropout)
    z_test = _project(ckpt, feats_test, cfg.geometric.dropout)
    bf = bona_fide_label(train_names_d)
    proto_order = [train_names_d[l] for l in sorted(train_names_d) if l != bf]
    res = analyze_unseen_pais(
        features_train=z_train, labels_train=y_train, train_names=train_names_d,
        features_unseen=z_test, labels_unseen=y_test, unseen_names=test_names_d,
        prototype_order=proto_order, bf_key=train_names_d[bf], tau=cfg.attribution.tau)
    return {u: r["ranked_anchors"] for u, r in res.items()}, proto_order


# ---------------------------------------------------------------------------
# Stability metrics vs identity
# ---------------------------------------------------------------------------

def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau-a over concordant/discordant pairs of two score vectors."""
    n = len(a)
    if n < 2:
        return 1.0
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    tot = n * (n - 1) / 2
    return (c - d) / tot if tot else 1.0


def compare(ref: Dict[str, List], other: Dict[str, List],
            proto_order: List[str]) -> Dict[str, Dict]:
    idx = {name: i for i, name in enumerate(proto_order)}
    out: Dict[str, Dict] = {}
    for u in ref:
        r_rank, o_rank = ref[u], other[u]
        r_top1, o_top1 = r_rank[0][0], o_rank[0][0]
        r_top2 = {r_rank[0][0], r_rank[1][0]}
        o_top2 = {o_rank[0][0], o_rank[1][0]}
        r_vec = np.array([dict(r_rank)[m] for m in proto_order])
        o_vec = np.array([dict(o_rank)[m] for m in proto_order])
        out[u] = {
            "top1_match": bool(r_top1 == o_top1),
            "top2_match": bool(r_top2 == o_top2),
            "kendall_tau": float(_kendall_tau(r_vec, o_vec)),
            "ref_top1": r_top1, "perm_top1": o_top1,
        }
    return out


# ---------------------------------------------------------------------------
# Permutations
# ---------------------------------------------------------------------------

def _load_cosine_optimal(cfg: Config, K: int) -> np.ndarray | None:
    """Read outputs/ordering_<scanner>.json if present -> permutation array."""
    p = Path("outputs") / f"ordering_{cfg.scanner}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    old = d.get("old_labels_in_order")          # old label placed at sector j (1..K order)
    if not old or len(old) != K:
        return None
    perm = np.zeros(K, dtype=int)
    for sector, old_label in enumerate(old, start=1):
        perm[int(old_label) - 1] = sector       # material old_label -> sector
    return perm


def build_permutations(cfg: Config, K: int, m_random: int, seed: int) -> Dict[str, np.ndarray]:
    perms: Dict[str, np.ndarray] = {"identity": np.arange(1, K + 1)}
    co = _load_cosine_optimal(cfg, K)
    if co is not None:
        perms["cosopt"] = co
    rng = np.random.default_rng(seed)
    seen = {tuple(v) for v in perms.values()}
    j = 0
    while j < m_random:
        cand = rng.permutation(np.arange(1, K + 1))
        if tuple(cand) in seen:
            continue
        seen.add(tuple(cand))
        perms[f"random{j:02d}"] = cand
        j += 1
    return perms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = base_parser("Permutation stability of the angular manifold.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-permutations", type=int, default=8,
                        help="Number of random permutations (plus identity + cosopt).")
    parser.add_argument("--perm-seed", type=int, default=12345,
                        help="RNG seed for generating the random permutations.")
    parser.add_argument("--train-seed", type=int, default=0,
                        help="Seed for the g_psi training runs.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override g_psi epochs (smoke tests).")
    parser.add_argument("--save-json", default=None)
    args = parser.parse_args()

    cfg = load_with_overrides(args)
    device_spec = args.device or cfg.device

    feats_train = np.load(features_path(cfg.paths.features_dir, cfg.scanner, "train"))
    feats_test = np.load(features_path(cfg.paths.features_dir, cfg.scanner, "test"))
    y_train = np.array(build_labels(
        read_split(split_path(cfg.paths.splits_dir, cfg.scanner, "train")), known_pairs(cfg)))
    y_test = np.array(build_labels(
        read_split(split_path(cfg.paths.splits_dir, cfg.scanner, "test")), unseen_pairs(cfg)))
    K = assert_contiguous_pai_labels(y_train.tolist())
    train_names_d, test_names_d = known_names(cfg), unseen_names(cfg)

    perms = build_permutations(cfg, K, args.num_permutations, args.perm_seed)
    LOG.info("=== permutation stability / scanner=%s K=%d permutations=%s ===",
             cfg.scanner, K, list(perms))

    attributions: Dict[str, Dict] = {}
    proto_order = None
    for name, perm in perms.items():
        ckpt = train_permuted_encoder(cfg, feats_train, y_train, perm, name,
                                      args.train_seed, device_spec, args.epochs)
        attributions[name], proto_order = attribution_for(
            ckpt, cfg, feats_train, y_train, feats_test, y_test,
            train_names_d, test_names_d)

    # Compare every permutation to identity.
    ref = attributions["identity"]
    per_perm: Dict[str, Dict] = {}
    top1_hits = top2_hits = taus = n = 0
    tau_sum = 0.0
    for name, attr in attributions.items():
        if name == "identity":
            continue
        cmp = compare(ref, attr, proto_order)
        per_perm[name] = cmp
        for u, m in cmp.items():
            top1_hits += m["top1_match"]
            top2_hits += m["top2_match"]
            tau_sum += m["kendall_tau"]
            n += 1

    summary = {
        "scanner": cfg.scanner, "K": K, "n_permutations": len(perms) - 1,
        "n_unseen": len(ref),
        "top1_match_rate": top1_hits / n if n else float("nan"),
        "top2_match_rate": top2_hits / n if n else float("nan"),
        "mean_kendall_tau": tau_sum / n if n else float("nan"),
    }

    print("\n" + "=" * 70)
    print(f"PERMUTATION STABILITY  scanner={cfg.scanner}  "
          f"perms={len(perms) - 1}  unseen={len(ref)}")
    print("=" * 70)
    print(f"  top-1 anchor match rate : {summary['top1_match_rate']:.3f}")
    print(f"  top-2 anchor match rate : {summary['top2_match_rate']:.3f}")
    print(f"  mean Kendall tau p_u,.  : {summary['mean_kendall_tau']:.3f}")
    print("  (1.000 = anchor semantics fully invariant to prototype permutation)")

    if args.save_json:
        out = Path(args.save_json)
        ensure_dir(out.parent)
        serial = {
            "summary": summary,
            "permutations": {k: v.tolist() for k, v in perms.items()},
            "attributions": {name: {u: [(m, float(w)) for m, w in ranked]
                                    for u, ranked in attr.items()}
                             for name, attr in attributions.items()},
            "per_permutation_vs_identity": per_perm,
        }
        out.write_text(json.dumps(serial, indent=2))
        LOG.info("saved -> %s", out)


if __name__ == "__main__":
    main()
