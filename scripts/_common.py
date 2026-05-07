"""Argparse and config-loading helpers used by the phase scripts.

This file is *not* part of the package; it lives next to the CLI scripts
to keep them self-contained while sharing boilerplate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

# Make the repo root importable when scripts are run as 'python scripts/...'.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xfpad.config import (  # noqa: E402
    Config,
    label_mapping_to_pairs,
    load_config,
    class_names_dict,
)


# ---------------------------------------------------------------------------
# Argparse boilerplate
# ---------------------------------------------------------------------------

def base_parser(description: str) -> argparse.ArgumentParser:
    """Create a parser with --config and --base-config options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", "-c", required=True, type=str,
        help="Path to a scanner YAML (e.g. configs/greenbit.yaml).",
    )
    parser.add_argument(
        "--base-config", default="configs/base.yaml", type=str,
        help="Path to configs/base.yaml (default: configs/base.yaml).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the seed from the config.",
    )
    return parser


def load_with_overrides(args: argparse.Namespace) -> Config:
    """Load the config and apply CLI overrides."""
    cfg = load_config(args.config, base_yaml=args.base_config)
    if getattr(args, "seed", None) is not None:
        cfg.seed = int(args.seed)
    return cfg


# ---------------------------------------------------------------------------
# Convenience accessors for the YAML sub-sections
# ---------------------------------------------------------------------------

def known_pairs(cfg: Config) -> Dict[tuple, int]:
    """Tuple-keyed mapping for the training labels."""
    return label_mapping_to_pairs(cfg.training_labels.mapping)


def unseen_pairs(cfg: Config) -> Dict[tuple, int]:
    """Tuple-keyed mapping for the unseen-PAI labels."""
    return label_mapping_to_pairs(cfg.unseen_labels.mapping)


def known_names(cfg: Config) -> Dict[int, str]:
    return class_names_dict(cfg.training_labels.names)


def unseen_names(cfg: Config) -> Dict[int, str]:
    return class_names_dict(cfg.unseen_labels.names)


def axes_lim(cfg: Config):
    al = cfg.plot.get("axes_lim")
    if al is None:
        return None
    return (tuple(al["xlim"]), tuple(al["ylim"]))


# ---------------------------------------------------------------------------
# Bona-fide-aware substring lookups
# ---------------------------------------------------------------------------

def bona_fide_label(names: Dict[int, str]) -> int:
    """Return the integer label that maps to 'Bona Fide' in the names dict."""
    for lbl, name in names.items():
        if name.lower().strip() in {"bona fide", "live"}:
            return int(lbl)
    raise ValueError("No bona fide label found in names dictionary.")
