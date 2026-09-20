"""Config loading and resolution (M01.1/M01.2).

YAML file -> validated dict. Model IDs and budgets change via config files,
never via source edits. Resolved configs are saved with each run (M01.2).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

try:
    import yaml
except ImportError as e:
    raise ImportError("pyyaml is required: pip install -r requirements.txt") from e

from .backends import validate_backend

DEFAULTS: dict = {
    "seed": 0,
    "resume": False,
    "backend": {"precision": "fp32", "decoder": "reference"},
    "data": {"names": [], "max_examples": 4},
    "budget": {"max_candidates": 8},
}


def load_config(path: str | Path) -> dict:
    """Parse a YAML config file; must be a mapping."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} must be a YAML mapping, got {type(cfg).__name__}")
    return cfg


def resolve_config(cfg: dict) -> dict:
    """Apply defaults and validate. Returns a new dict; input untouched."""
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping, got {type(cfg).__name__}")
    resolved = copy.deepcopy(DEFAULTS)
    for key, value in cfg.items():
        if isinstance(value, dict) and isinstance(resolved.get(key), dict):
            resolved[key].update(value)
        else:
            resolved[key] = value
    model_id = resolved.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("config needs a non-empty string 'model_id'")
    if not isinstance(resolved.get("seed"), int):
        raise ValueError("config 'seed' must be an int")
    resolved["backend"] = {
        "precision": validate_backend(resolved.get("backend", {})).precision,
        "decoder": validate_backend(resolved.get("backend", {})).decoder,
    }
    if int(resolved["data"].get("max_examples", 0)) <= 0:
        raise ValueError("config 'data.max_examples' must be > 0")
    if int(resolved["budget"].get("max_candidates", 0)) <= 0:
        raise ValueError("config 'budget.max_candidates' must be > 0")
    return resolved


def config_hash(cfg: dict) -> str:
    """Stable short hash of the resolved config for run manifests."""
    canonical = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def save_resolved(cfg: dict, path: str | Path) -> None:
    """Write the resolved config next to run artifacts (M01.2)."""
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
