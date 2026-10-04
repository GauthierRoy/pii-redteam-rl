"""Build dataset-derived SFT pairs from a local AI4Privacy sample (M02).

Usage: uv run python scripts/build_sft_pairs.py --in artifacts/ai4privacy_sample.jsonl
Writes artifacts/sft_pairs_sample.jsonl; reports quarantine/selection counts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pii_redteam.data import QuarantinedRecord, build_sft_pair, is_single_person_once, to_canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="artifacts/ai4privacy_sample.jsonl")
    parser.add_argument("--out", default="artifacts/sft_pairs_sample.jsonl")
    parser.add_argument("--max-pairs", type=int, default=8)
    args = parser.parse_args()
    n_rows = n_quarantined = n_multi = 0
    pairs = []
    with open(args.inp) as f:
        for line in f:
            if not line.strip():
                continue
            n_rows += 1
            try:
                canonical = to_canonical(json.loads(line))
            except QuarantinedRecord:
                n_quarantined += 1
                continue
            if not is_single_person_once(canonical):
                n_multi += 1
                continue
            if len(pairs) >= args.max_pairs:
                continue
            request, target = build_sft_pair(canonical, request_id=f"sft-{len(pairs):04d}")
            pairs.append({"request": request, "target": target})
    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(
        f"rows={n_rows} quarantined={n_quarantined} multi-or-repeat={n_multi} "
        f"pairs={len(pairs)} -> {args.out}"
    )


if __name__ == "__main__":
    main()
