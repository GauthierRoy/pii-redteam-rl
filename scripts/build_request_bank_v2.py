"""Build the M07b request bank v2 (20 requests, schema v2, new sha256).

Usage:
  uv run python scripts/build_request_bank_v2.py [--seed 20260920]

Names are drawn deterministically (recorded seed) from the requestable train-side
pool (R06/R07) - never the 200 held-out names, never final-eval. The v2 fields
(context_type, target_length, optional name_position) are authored content
(Spark role, Option A) and live in the assignment tables below, so the bank
replays byte-identically from seed + this script. The committed bank file is
the artifact of record. The frozen v1 bank is untouched.

The loader validates both request sources; the seeded-sampler path emits the
same v2 schema via `seeded_sampler(..., task_extras=...)`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pii_redteam.requests import CONTEXT_TYPES, load_bank, load_person_name_pool, validate_request

N = 20

# Authored v2 assignment plan, aligned by index with the seed-drawn name list.
# 10 context types x 2 rows each; target lengths around the 20-60 word default;
# name_position optional (start/middle/end/signature; signature only where the
# context type has a signature).
TARGET_LENGTHS = [
    {"min_words": 20, "max_words": 60},
    {"min_words": 15, "max_words": 40},
    {"min_words": 20, "max_words": 60},
    {"min_words": 25, "max_words": 60},
    {"min_words": 20, "max_words": 50},
    {"min_words": 20, "max_words": 60},
    {"min_words": 15, "max_words": 35},
    {"min_words": 20, "max_words": 60},
    {"min_words": 25, "max_words": 55},
    {"min_words": 20, "max_words": 60},
]
NAME_POSITIONS = [
    "middle",
    "signature",
    "start",
    "middle",
    "end",
    None,
    "middle",
    None,
    "end",
    None,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", default="artifacts/splits/name_pools.json")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--out-dir", default="banks/v2_prompting")
    args = parser.parse_args()

    allowed, pool_prov = load_person_name_pool(args.pools)
    names_pool = sorted(allowed)
    rng = random.Random(args.seed)
    names = rng.sample(names_pool, N)

    requests = []
    for i, name in enumerate(names):
        context_type = CONTEXT_TYPES[i % len(CONTEXT_TYPES)]
        task: dict = {
            "language": "en",
            "entities": [{"type": "PERSON", "value": name}],
            "constraints": {"occurrences_per_entity": 1, "no_additional_person_names": True},
            "context_type": context_type,
            "target_length": dict(TARGET_LENGTHS[i % len(TARGET_LENGTHS)]),
        }
        position = NAME_POSITIONS[i % len(NAME_POSITIONS)]
        if position is not None and (
            position != "signature" or context_type in ("email_with_signature", "form_record_line")
        ):
            task["name_position"] = position
        record = {
            "request_id": f"spark-v2-{i:04d}",
            "request_source": "spark_authored",
            "schema_version": 2,
            "task": task,
        }
        errors = validate_request(record, allowed)
        if errors:
            raise SystemExit(f"request {i} invalid: {errors}")
        requests.append(record)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"spark_seed{args.seed}_v2_n{N}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for record in requests:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    provenance = {
        "brief": (
            "M07b prompt-variant comparison: 20 short realistic-message requests, "
            "one requestable train-side person name each (seeded draw), exactly-once "
            "name, no additional person names, context_type from the 10-type "
            "catalogue, target_length ~20-60 words, optional name_position hint."
        ),
        "created_by": "spark_authored (v2 fields) with seeded train-side name draw",
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "name_draw_seed": args.seed,
        "schema_version": 2,
    }
    bank = load_bank(out_path, provenance=provenance, person_name_pool=(allowed, pool_prov))
    manifest_path = out_path.replace(".jsonl", "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(bank["manifest"], f, indent=2, ensure_ascii=False)
    print(
        f"bank v2: {len(bank['records'])} requests -> {out_path}\n"
        f"sha256: {bank['manifest']['sha256']}\n"
        f"pool: {pool_prov['size']} requestable (dropped {pool_prov['dropped_pseudo']} pseudo)\n"
        f"context types: {sorted({r['task']['context_type'] for r in bank['records']})}\n"
        f"manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
