"""M06.2/M06.3: turn the base attacker's candidate pool into the plain and generate-and-filter sets.

Input: the `rl-generate` run directory of configs/m06_base_generate.yaml (accepted.jsonl,
rl_generate_report.json). Output: data/synthetic/plain_aug_v1.jsonl and gf_aug_v1.jsonl with
manifests (selection rule, counts, target-detector misses).

    python scripts/finalize_base_aug_v1.py --run-dir <path>/m06-base-generate-<stamp>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter

from pii_redteam.baselines import (
    ARM_ROWS,
    HARD_CONTEXT_CAP,
    aug_row,
    hard_selection,
    plain_selection,
)

ARMS = {
    "plain_aug_v1": (plain_selection, "first sample per request, accepted, in request order"),
    "gf_aug_v1": (
        hard_selection,
        f"lowest target-detector p on the name, one per name, at most {HARD_CONTEXT_CAP} per "
        "context type",
    ),
}


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    accepted_path = os.path.join(args.run_dir, "accepted.jsonl")
    with open(accepted_path, encoding="utf-8") as f:
        accepted = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(args.run_dir, "rl_generate_report.json"), encoding="utf-8") as f:
        report = json.load(f)
    for dataset, (select, rule) in ARMS.items():
        chosen = select(accepted, ARM_ROWS)
        prefix = dataset.replace("_", "-")
        rows = [
            aug_row(prefix, i, row, f"base_attacker:{report['run_id']}")
            for i, row in enumerate(chosen)
        ]
        out = f"data/synthetic/{dataset}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest = {
            "file": out,
            "sha256": sha256(out),
            "rows": len(rows),
            "names": len({r["request"]["entities"][0]["value"] for r in rows}),
            "source": {
                "run_id": report["run_id"],
                "accepted_sha256": sha256(accepted_path),
                "pool": report["counts"],
            },
            "attacker": "base Qwen3.5-2B (no adapter)",
            "filters": "validity v2, person judge (mistral-medium-2604), audit other_person "
            "false, no exact repeat",
            "selection": rule,
            "target_detector_misses": sum(r["target_detector_p_name"] < 0.5 for r in rows),
            "by_context": dict(Counter(r["request"]["context_type"] for r in rows)),
            "opens_with_dear": sum(r["target"]["text"].lower().startswith("dear") for r in rows),
        }
        with open(out.replace(".jsonl", "_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
