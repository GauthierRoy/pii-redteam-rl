"""Re-label a samples.jsonl (M07a probe or M07b comparison) under v2 validity.

Usage:
  uv run python scripts/relabel_v2.py --samples <path> [--out <path>]

Applies the strict v2 rules (validity_v2) to every sample, keeps v1 labels and
p_target for comparison, and prints:
  - v1 -> v2 validity counts
  - every valid->invalid flip with its v2 failure reason
  - the hard panel (lowest p_target) and which hard samples become invalid

Works on raw M07a probe samples (fields: raw_text, name, truncated, status,
p_target_mean) and on M07b samples_raw.jsonl (the runner already stores v2
labels; this script recomputes them independently).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pii_redteam.validity_v2 import relabel_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="path to a samples JSONL file")
    parser.add_argument("--out", default=None, help="optional path for the relabeled JSONL")
    args = parser.parse_args()

    with open(args.samples, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    report = relabel_samples(samples)

    print(f"samples: {report['n']}")
    print(f"v1 valid: {report['v1_valid']}  ->  v2 valid: {report['v2_valid']}")
    print(f"\nvalid -> invalid flips: {len(report['flips_valid_to_invalid'])}")
    for flip in report["flips_valid_to_invalid"]:
        p = flip["p_target_mean"]
        p_str = f"p_target={p:.3f}" if isinstance(p, (int, float)) else "p_target=?"
        print(
            f"  [{flip['group_id']} #{flip['sample_idx']}] {p_str} {flip['outcome']} -> "
            f"{flip['status_v2']}\n    {flip['text']!r}"
        )
    print("\nhard panel (lowest p_target), v2 outcome:")
    for h in report["hard_samples"]:
        p = h["p_target_mean"]
        p_str = f"p_target={p:.3f}" if isinstance(p, (int, float)) else "p_target=?"
        print(
            f"  [{h['group_id']} #{h['sample_idx']}] {p_str} {h['outcome']} "
            f"v1={h['status_v1']} -> v2={h['status_v2']}\n    {h['text']!r}"
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for s in report["samples"]:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"\nrelabeled samples -> {args.out}")


if __name__ == "__main__":
    main()
