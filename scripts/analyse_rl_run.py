"""Readout of one RL run directory: training curve per block of steps and held-out pre/post.

Reads train_texts.jsonl (one row per completion, with `reward_call` = step), samples_pre.jsonl
and samples_post.jsonl. Probabilities are whatever the run's scorer wrote (`p_target_mean`;
after D25 the PERSON / not-PERSON reading). A miss is p < 0.5.

Usage: python scripts/analyse_rl_run.py RUN_DIR [--block 50] [--examples 8]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics


def text_of(raw: str) -> str:
    match = re.search(r'"text"\s*:\s*"(.*)', raw or "", re.S)
    return (match.group(1) if match else raw or "").rstrip('"} \n')


def opening(text: str) -> str:
    words = text.split()
    return words[0].lower().strip(",.!:") if words else ""


def block_rows(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["status"] == "valid" and r.get("p_target_mean") is not None]
    ps = [r["p_target_mean"] for r in valid]
    statuses = collections.Counter(r["status"] for r in rows)
    return {
        "texts": len(rows),
        "valid": len(valid),
        "mean_p": round(statistics.mean(ps), 4) if ps else None,
        "median_p": round(statistics.median(ps), 5) if ps else None,
        "misses": sum(p < 0.5 for p in ps),
        "mean_reward": round(statistics.mean(r["reward"] for r in rows), 3) if rows else None,
        "top_invalid": statuses.most_common(4)[1:] if statuses else [],
        "top_openings": collections.Counter(
            opening(text_of(r["raw_text"])) for r in valid
        ).most_common(3),
    }


def heldout(path: str) -> dict:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    valid = [r for r in rows if r.get("status") == "valid" and r.get("p_target_mean") is not None]
    ps = [r["p_target_mean"] for r in valid]
    return {
        "samples": len(rows),
        "valid": len(valid),
        "misses": sum(p < 0.5 for p in ps),
        "median_p": round(statistics.median(ps), 5) if ps else None,
        "other_person": sum(1 for r in valid if r.get("other_person")),
        "top_openings": collections.Counter(
            opening(text_of(r["raw_text"])) for r in valid
        ).most_common(4),
        "statuses": collections.Counter(r.get("status") for r in rows).most_common(6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--block", type=int, default=50)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    train = [
        json.loads(line)
        for line in open(os.path.join(args.run_dir, "train_texts.jsonl"), encoding="utf-8")
    ]
    by_block: dict[int, list[dict]] = collections.defaultdict(list)
    for row in train:
        by_block[row["reward_call"] // args.block].append(row)
    print(f"training, blocks of {args.block} steps")
    for block in sorted(by_block):
        print(f"  {block * args.block:>4}", json.dumps(block_rows(by_block[block])))
    for name in ("pre", "post"):
        path = os.path.join(args.run_dir, f"samples_{name}.jsonl")
        if os.path.exists(path):
            print(name, json.dumps(heldout(path)))
    last = max(by_block) if by_block else 0
    late_misses = [
        r
        for r in train
        if r["reward_call"] // args.block >= last - 1
        and r["status"] == "valid"
        and (r.get("p_target_mean") or 1) < 0.5
    ]
    print(f"late training misses: {len(late_misses)}; examples:")
    for row in late_misses[: args.examples]:
        print(f"  {row['p_target_mean']:.2f} {row['name']} | {text_of(row['raw_text'])[:170]}")


if __name__ == "__main__":
    main()
