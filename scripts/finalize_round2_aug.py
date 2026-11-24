"""M10 round 2: augmentation sets against the RL-retrained detector, matched at 316 rows.

Inputs: the `rl-generate` run directories of configs/m10_round2_rl_generate.yaml (the 1000-step
person attacker, one sample per request) and configs/m10_round2_base_generate.yaml (the base
attacker, four samples per request), both on the same 600 requests and scored by the
RL-retrained detector. Outputs, with manifests:
  rl_aug_v2    = the RL attacker's accepted texts in request order (first 316)
  gf_aug_v2    = base pool, lowest retrained-detector p first, one per name, <= 48 per type
  plain_aug_v2 = base pool, accepted first samples in request order

    python scripts/finalize_round2_aug.py --rl-run-dir <dir> --base-run-dir <dir>
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
from pii_redteam.training.opening_share import opening_word

DETECTOR = "retrained-rl-pilot-20261115-epoch3 (4755db73...)"


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load(run_dir: str) -> tuple[list[dict], dict, str]:
    path = os.path.join(run_dir, "accepted.jsonl")
    with open(path, encoding="utf-8") as f:
        accepted = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(run_dir, "rl_generate_report.json"), encoding="utf-8") as f:
        report = json.load(f)
    return accepted, report, path


def name_first(row: dict) -> bool:
    """The text starts with the requested name (the round-1 attacker's pattern)."""
    return row["target"]["text"].lstrip(" \"'*").startswith(row["request"]["entities"][0]["value"])


def write_arm(dataset: str, chosen: list[dict], provenance: str, source: dict, extra: dict) -> None:
    prefix = dataset.replace("_", "-")
    rows = [aug_row(prefix, i, row, provenance) for i, row in enumerate(chosen)]
    out = f"data/synthetic/{dataset}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    openings = Counter(
        opening_word(r["target"]["text"], r["request"]["entities"][0]["value"]) for r in rows
    )
    manifest = {
        "file": out,
        "sha256": sha256(out),
        "rows": len(rows),
        "names": len({r["request"]["entities"][0]["value"] for r in rows}),
        "source": source,
        "filters": "validity v2, person judge (mistral-medium-2604), audit other_person false, "
        "no exact repeat",
        "scored_by": DETECTOR,
        "detector_misses": sum(r["target_detector_p_name"] < 0.5 for r in rows),
        "name_first": sum(name_first(r) for r in rows),
        "top_openings_after_name": dict(openings.most_common(8)),
        "by_context": dict(Counter(r["request"]["context_type"] for r in rows)),
        **extra,
    }
    with open(out.replace(".jsonl", "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-run-dir", required=True)
    parser.add_argument("--base-run-dir", required=True)
    args = parser.parse_args()

    rl_accepted, rl_report, rl_path = load(args.rl_run_dir)
    rl_source = {
        "run_id": rl_report["run_id"],
        "accepted_sha256": sha256(rl_path),
        "pool": rl_report["counts"],
    }
    write_arm(
        "rl_aug_v2",
        plain_selection(rl_accepted, ARM_ROWS),
        f"rl_attacker:{rl_report['run_id']}",
        rl_source,
        {
            "attacker": "m10-attack-person-modal-20261124-064157 adapter (1000 steps)",
            "selection": "accepted texts in request order",
        },
    )

    base_accepted, base_report, base_path = load(args.base_run_dir)
    base_source = {
        "run_id": base_report["run_id"],
        "accepted_sha256": sha256(base_path),
        "pool": base_report["counts"],
    }
    for dataset, select, rule in (
        (
            "gf_aug_v2",
            hard_selection,
            f"lowest retrained-detector p on the name, one per name, <= {HARD_CONTEXT_CAP} per type",
        ),
        ("plain_aug_v2", plain_selection, "first sample per request, accepted, in request order"),
    ):
        write_arm(
            dataset,
            select(base_accepted, ARM_ROWS),
            f"base_attacker:{base_report['run_id']}",
            base_source,
            {"attacker": "base Qwen3.5-2B (no adapter)", "selection": rule},
        )
    pool_name_first = sum(r["text"].lstrip(" \"'*").startswith(r["name"]) for r in base_accepted)
    print(f"base pool: {pool_name_first}/{len(base_accepted)} accepted texts open with the name")


if __name__ == "__main__":
    main()
