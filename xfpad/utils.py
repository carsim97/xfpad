"""Cross-cutting utilities (seeding, device resolution, path helpers, logging).

Torch is imported lazily so that path / I/O helpers can be used without
installing torch (e.g. when only label-mapping or YAML config logic is needed).
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Make torch / numpy / python random deterministic-ish."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str = "auto"):
    """Return a torch.device. Lazy import so the helper is optional."""
    import torch

    if spec == "cuda":
        return torch.device("cuda")
    if spec == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Standard repo paths
# ---------------------------------------------------------------------------

def split_path(splits_dir: str | Path, scanner: str, mode: str) -> Path:
    """data/splits/<scanner>_<mode>.txt   (mode in {'train','test'})"""
    return Path(splits_dir) / f"{scanner}_{mode}.txt"


def features_path(features_dir: str | Path, scanner: str, mode: str) -> Path:
    """data/features/<scanner>_<mode>.npy"""
    return Path(features_dir) / f"{scanner}_{mode}.npy"


def feature_extractor_ckpt(ckpt_dir: str | Path, scanner: str) -> Path:
    return Path(ckpt_dir) / f"feature_extractor_{scanner}.pth"


def geometric_ckpt(ckpt_dir: str | Path, scanner: str, seed: int) -> Path:
    return Path(ckpt_dir) / f"geometric_{scanner}_{seed}.pth"


def pad_detector_ckpt(ckpt_dir: str | Path, scanner: str,
                      backbone: str, ablation: str, run: int) -> Path:
    """e.g. checkpoints/pad/greenbit/baseline/mobilenet_v2_run3.pth"""
    return (Path(ckpt_dir) / "pad" / scanner / ablation
            / f"{backbone}_run{run}.pth")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_split(path: str | Path) -> list[str]:
    """Read a split file (one path per line, blank lines skipped)."""
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "xfpad", level: int = logging.INFO) -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False
    _LOGGERS[name] = logger
    return logger
