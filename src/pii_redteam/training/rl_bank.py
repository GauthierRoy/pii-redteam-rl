"""RL request bank (M08.3): seeded v2 requests, train-side names only, disjoint from banks.

Names come from the requestable train-side pool (R06/R07), never the held-out names, and
exclude every name already used by the listed comparison banks, so RL requests never overlap
the requests later used to compare against the base model.
"""

from __future__ import annotations

import json
import random

from ..requests import CONTEXT_TYPES, seeded_sampler, validate_request

TARGET_LENGTHS = ((15, 35), (20, 50), (20, 60), (25, 60))
NAME_POSITIONS = (None, "start", "middle", "end", "signature")
SIGNATURE_CONTEXTS = ("email_with_signature", "form_record_line")


def bank_names(path: str) -> frozenset[str]:
    """Every PERSON value requested in a JSONL bank."""
    names = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                names.update(e["value"] for e in json.loads(line)["task"]["entities"])
    return frozenset(names)


def build_rl_requests(
    pool: frozenset[str],
    *,
    n: int,
    seed: int,
    exclude: frozenset[str] = frozenset(),
    prefix: str = "rl",
) -> list[dict]:
    """`n` schema-v2 requests with balanced context types and seeded names/lengths/positions."""
    available = sorted(pool - exclude)
    if n > len(available):
        raise ValueError(f"asked for {n} names, only {len(available)} available after exclusions")
    rng = random.Random(seed)
    names = rng.sample(available, n)
    extras = []
    for i in range(n):
        context_type = CONTEXT_TYPES[i % len(CONTEXT_TYPES)]
        min_words, max_words = rng.choice(TARGET_LENGTHS)
        task = {
            "context_type": context_type,
            "target_length": {"min_words": min_words, "max_words": max_words},
        }
        position = rng.choice(NAME_POSITIONS)
        if position is not None and (position != "signature" or context_type in SIGNATURE_CONTEXTS):
            task["name_position"] = position
        extras.append(task)
    requests = seeded_sampler(
        names, seed=seed, prefix=prefix, allowed_names=pool, task_extras=extras
    )
    for request in requests:
        errors = validate_request(request, pool)
        if errors:
            raise ValueError(f"{request['request_id']}: {errors}")
    return requests
