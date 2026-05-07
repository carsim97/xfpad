"""YAML configuration loading with deep merge.

Scanner configs (e.g. configs/greenbit.yaml) are layered on top of
configs/base.yaml so that shared hyperparameters live in one place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


# ---------------------------------------------------------------------------
# Attribute-access dict
# ---------------------------------------------------------------------------

class Config(dict):
    """Dict that also supports cfg.foo.bar attribute access."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive merge: nested dicts are merged, scalars/lists are overwritten."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_config(scanner_yaml: str | Path,
                base_yaml: str | Path = "configs/base.yaml") -> Config:
    """Load a scanner config layered on top of the base config.

    Parameters
    ----------
    scanner_yaml : path to e.g. configs/greenbit.yaml
    base_yaml    : path to configs/base.yaml (default: relative to CWD)

    Returns
    -------
    Config (attribute-accessible nested dict).
    """
    base_path = Path(base_yaml)
    scanner_path = Path(scanner_yaml)

    with base_path.open("r") as f:
        base = yaml.safe_load(f) or {}
    with scanner_path.open("r") as f:
        override = yaml.safe_load(f) or {}

    merged = _deep_merge(base, override)
    return _wrap(merged)


# ---------------------------------------------------------------------------
# Label-mapping helpers
# ---------------------------------------------------------------------------

def label_mapping_to_pairs(mapping: List[Dict[str, Any]]) -> Dict[Tuple[str, ...], int]:
    """Convert a YAML 'mapping' list to the tuple-keyed dict expected by build_labels.

    YAML form
        - { match: ["Live"],            label: 0 }
        - { match: ["VBN", "Consensual"], label: 1 }

    Returns
    -------
    {("Live",): 0, ("VBN", "Consensual"): 1, ...}
    """
    out: Dict[Tuple[str, ...], int] = {}
    for entry in mapping:
        match = tuple(entry["match"])
        label = int(entry["label"])
        out[match] = label
    return out


def class_names_dict(names: Dict[Any, str]) -> Dict[int, str]:
    """Normalize YAML name keys (which may be parsed as str) to int."""
    return {int(k): str(v) for k, v in names.items()}
