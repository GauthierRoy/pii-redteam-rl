"""Split roles, duplicate checks, and name pools for the AI4Privacy pilot (M02.3–M02.4).

Pure stdlib helpers. I/O and manifest writing live in scripts/carve_splits.py; the
policies applied here are documented in docs/DATA_CARD.md.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter


def text_sha256(text: str) -> str:
    """SHA-256 of the exact text (UTF-8), used for exact-duplicate and lock manifests."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def template_skeleton(text: str, masks: list[dict]) -> str:
    """Replace every annotated PII span with '<LABEL>' to expose the underlying template.

    The dataset is synthetic; rows generated from the same template with different
    values share this skeleton. Skeleton equality is reported as template-level
    near-duplication, not banned (see docs/DATA_CARD.md).
    """
    out = text
    for m in sorted(masks, key=lambda m: m["start"], reverse=True):
        out = out[: m["start"]] + f"<{m['label']}>" + out[m["end"] :]
    return out


def dedup_by_text(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Keep the first occurrence of each distinct ``text``.

    Returns ``(kept_rows, duplicate_example_ids)`` in input order.
    """
    seen: set[str] = set()
    kept: list[dict] = []
    dupes: list[str] = []
    for row in rows:
        digest = text_sha256(row["text"])
        if digest in seen:
            dupes.append(row["example_id"])
        else:
            seen.add(digest)
            kept.append(row)
    return kept, dupes


def carve_final_eval(example_ids: list[str], *, seed: int, n: int) -> list[str]:
    """Deterministic seeded selection of ``n`` example ids for the final-eval partition."""
    rng = random.Random(seed)
    ordered = sorted(example_ids)
    rng.shuffle(ordered)
    return ordered[:n]


def person_name_counts(canonical_rows: list[dict]) -> Counter:
    """Frequency of every PERSON value across rows (full merged spans)."""
    counts: Counter = Counter()
    for row in canonical_rows:
        for span in row["spans"]:
            counts[row["text"][span["start"] : span["end"]]] += 1
    return counts


def carve_heldout_names(
    final_eval_names: Counter, train_side_names: Counter, *, seed: int, heldout_n: int
) -> list[str]:
    """Seeded pick of names occurring only in final-eval rows (never in train-side rows).

    Returns at most ``heldout_n`` sorted names; disjoint from ``train_side_names`` by
    construction.
    """
    unique = sorted(name for name in final_eval_names if name not in train_side_names)
    rng = random.Random(seed + 1)
    rng.shuffle(unique)
    return sorted(unique[:heldout_n])
