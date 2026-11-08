"""Durable run outputs: samples reach disk as they are generated, not at the end."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable


def append_jsonl(path: str, rows: Iterable[dict]) -> None:
    """Append rows and force them to disk, so a crash or disconnect loses at most one batch."""
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    """Replace a JSONL file atomically, so a crash mid-write keeps the previous version."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
