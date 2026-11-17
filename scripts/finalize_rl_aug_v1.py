"""Turn the accepted texts of an RL generation run into a detector augmentation file.

Input: `accepted.jsonl` from `rl-generate` (Modal volume, run m09-rl-generate-20261115-114151).
Output: `data/synthetic/rl_aug_v1.jsonl` in the same row format as `target_detector_aug_v1`
(one PERSON entity, its character span) plus a manifest with hashes and counts.

    python scripts/finalize_rl_aug_v1.py --accepted <path>/accepted.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json

RUN_ID = "m09-rl-generate-20261115-114151"
ATTACKER = "long-20261114-004948 (D18 reward, 840 steps)"
OUT = "data/synthetic/rl_aug_v1.jsonl"


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def to_aug_row(index: int, row: dict) -> dict:
    start, end = row["name_span"]
    if row["text"][start:end] != row["name"]:
        raise ValueError(f"{row['request_id']}: span is not the name")
    return {
        "id": f"rl-aug-v1-{index:05d}",
        "split": "augmentation",
        "request": {
            "language": "en",
            "context_type": row["context_type"],
            "entities": [{"type": "PERSON", "value": row["name"]}],
        },
        "target": {"text": row["text"], "span": [start, end]},
        "provenance": f"rl_attacker:{RUN_ID}",
        "source_request_id": row["request_id"],
        "target_detector_p_name": row["p_target_mean"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted", required=True)
    args = parser.parse_args()
    with open(args.accepted, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    out_rows = [to_aug_row(i, row) for i, row in enumerate(rows)]
    with open(OUT, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "file": OUT,
        "sha256": sha256(OUT),
        "rows": len(out_rows),
        "names": len({r["request"]["entities"][0]["value"] for r in out_rows}),
        "source": {"run_id": RUN_ID, "accepted_sha256": sha256(args.accepted)},
        "attacker": ATTACKER,
        "filters": "validity v2, person judge (mistral-medium-2604), audit other_person false, "
        "no exact repeat",
        "target_detector_misses": sum(r["target_detector_p_name"] < 0.5 for r in out_rows),
        "opens_with_dear": sum(r["target"]["text"].lower().startswith("dear") for r in out_rows),
    }
    with open(OUT.replace(".jsonl", "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
