"""Build a frozen Option B (seeded sampler) generation-request bank (M05).

Usage:
  uv run python scripts/build_request_bank.py --n 200 --seed 20260920

Draws `--n` requestable names from the train-side pool (R06 + R07 enforced by the
loader), emits requests through the shared schema, writes the bank JSONL, then
re-imports it through `load_bank` with the pool attached as a validation + provenance
gate. The saved bank is the reproducibility artifact: same seed + pool version ->
same bank; never assume regeneration.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pii_redteam.requests import load_bank, load_person_name_pool, seeded_sampler, validate_request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", default="artifacts/splits/name_pools.json")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--out-dir", default="artifacts/request_bank")
    args = parser.parse_args()

    allowed, pool_prov = load_person_name_pool(args.pools)
    names = sorted(allowed)
    if args.n > len(names):
        raise SystemExit(f"--n={args.n} exceeds requestable pool size {len(names)}")
    rng = random.Random(args.seed)
    chosen = rng.sample(names, args.n)
    requests = seeded_sampler(chosen, seed=args.seed, allowed_names=allowed)
    for i, request in enumerate(requests):
        errors = validate_request(request, allowed)
        if errors:
            raise SystemExit(f"request {i} invalid: {errors}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"sampler_seed{args.seed}_n{args.n}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    provenance = {
        "brief": (
            "Option B programmatic sampler: short English generation requests, one "
            "requestable train-side person name per request, name exactly once, "
            "no additional person names."
        ),
        "created_by": "seeded_sampler",
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    bank = load_bank(out_path, provenance=provenance, person_name_pool=(allowed, pool_prov))
    manifest_path = out_path.replace(".jsonl", "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(bank["manifest"], f, indent=2, ensure_ascii=False)
    print(
        f"bank: {len(bank['records'])} requests -> {out_path}\n"
        f"sha256: {bank['manifest']['sha256']}\n"
        f"pool: {pool_prov['size']} requestable (dropped {pool_prov['dropped_pseudo']} pseudo)\n"
        f"manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
