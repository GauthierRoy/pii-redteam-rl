"""Run manifests and the checkpoint-resume guard (M01.2/M01.3).

Every run writes manifest.json. Resuming into a directory that holds a run from a
different model or config fails loudly instead of silently restarting.
"""

from __future__ import annotations

import datetime
import json
import platform
import sys
import uuid
from pathlib import Path

MANIFEST_NAME = "manifest.json"


class ResumeError(RuntimeError):
    """Raised when a run directory cannot be safely resumed."""


def new_manifest(*, model_id: str, config_hash: str, seed: int) -> dict:
    """Create a fresh manifest (status 'running')."""
    return {
        "run_id": uuid.uuid4().hex[:12],
        "model_id": model_id,
        "config_hash": config_hash,
        "seed": seed,
        "status": "running",
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "counters": {},
    }


def manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / MANIFEST_NAME


def save_manifest(manifest: dict, run_dir: str | Path) -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with open(manifest_path(run_dir), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def load_manifest(run_dir: str | Path) -> dict | None:
    """Return the stored manifest, None when the dir holds no run, ResumeError if corrupt."""
    path = manifest_path(run_dir)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ResumeError(f"corrupt manifest at {path}: {e}") from e


def guard_resume(
    run_dir: str | Path, *, model_id: str, config_hash: str, resume: bool
) -> dict | None:
    """Enforce the resume policy; returns the existing manifest when resuming, else None."""
    existing = load_manifest(run_dir)
    if existing is None:
        return None
    if not resume:
        raise ResumeError(
            f"run dir {run_dir} already holds run {existing.get('run_id')} "
            f"(model {existing.get('model_id')}); use resume=true or a fresh --out dir"
        )
    if existing.get("model_id") != model_id:
        raise ResumeError(
            "refusing resume: existing run model "
            f"{existing.get('model_id')!r} != requested {model_id!r} "
            "(would silently restart from an unrelated model)"
        )
    if existing.get("config_hash") != config_hash:
        raise ResumeError(
            "refusing resume: config changed since the stored run "
            "(would silently lose accounting data); use a fresh --out dir"
        )
    return existing


def complete_manifest(
    run_dir: str | Path, *, status: str = "done", counters: dict | None = None
) -> dict:
    """Mark the stored manifest finished with final counters."""
    manifest = load_manifest(run_dir)
    if manifest is None:
        raise ResumeError(f"no manifest to complete in {run_dir}")
    manifest["status"] = status
    manifest["counters"] = counters or {}
    save_manifest(manifest, run_dir)
    return manifest
