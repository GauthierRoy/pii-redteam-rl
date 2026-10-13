"""Re-score saved comparison samples: D15 labels + official probe scorer + position compliance.

Usage (needs the ml group):
  PYTHONPATH=src uv run --group ml python scripts/rescore_v2.py \
      --samples results/m07b_prompt_v2/m07b-20261012-224819/samples_raw.jsonl \
      --detector /tmp/d0zip/pii-redteam-rl/d0-20261004-142714/d0-best \
      --detector-sha 0314ebe36d7f4dc6d784f0e139ff5b9d32633aef1a6f88e764a610a8d8ec43cb

Applies the tightened (D15) validity-v2 rules to every saved raw sample, scores
all valid ones with the OFFICIAL probe scorer (`pii_redteam.probe._score_with_d0`
— verified against the D0 training eval pipeline by the M07a scorer-sanity
regression, PR #5), computes per-hint name-position compliance, and reports
per-cell reward metrics. Also cross-checks the run's local reference-replica
p_target values against the official scorer on the common samples.

Outputs, next to the samples file: samples_rescored.jsonl (all 960, D15 labels
+ official scores) and rescore_report.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from typing import Any, cast

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pii_redteam.probe import _score_with_d0
from pii_redteam.probe_metrics import reward_metrics
from pii_redteam.prompt_v2 import hint_compliance, name_position_bin, opens_with_name
from pii_redteam.validity_v2 import classify_sample_v2


def _sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--detector-sha", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--bank", required=True, help="v2 bank JSONL for name_position hints")
    parser.add_argument("--bank-sha", required=True)
    args = parser.parse_args()

    if _sha256(args.bank) != args.bank_sha:
        raise SystemExit("bank lock mismatch")
    hints = {
        r["request_id"]: r["task"].get("name_position")
        for r in (json.loads(line) for line in open(args.bank, encoding="utf-8") if line.strip())
    }

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    actual_sha = _sha256(os.path.join(args.detector, "model.safetensors"))
    if actual_sha != args.detector_sha:
        raise SystemExit(f"D0 lock mismatch: {actual_sha} != {args.detector_sha}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d0_tok = AutoTokenizer.from_pretrained(args.detector)
    d0 = AutoModelForTokenClassification.from_pretrained(args.detector).to(device)
    d0.eval()
    print(f"official probe scorer ready (frozen D0 sha {actual_sha[:12]}…, {device})")

    with open(args.samples, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    print(f"samples: {len(samples)}")

    # ---- D15 re-label
    for s in samples:
        s["name_position_hint"] = hints.get(s["request_id"])
        s["status_asrun"] = s.get("status")
        s["p_target_replica"] = s.get("p_target_mean")
        v2 = classify_sample_v2(s["raw_text"], s["name"], s["truncated"])
        s.update(v2)  # status/text/target_span now D15 labels

    # ---- official scorer over D15-valid samples
    to_score = [
        {"text": s["text"], "target_span": s["target_span"], "status": "valid"}
        for s in samples
        if s["status"] == "valid"
    ]
    _score_with_d0(d0, d0_tok, to_score, args.max_length)
    scored = {id(s): s for s in to_score}
    i = 0
    replica_diffs = []
    for s in samples:
        if s["status"] != "valid":
            s.pop("p_target_mean", None)
            s.pop("p_target_min", None)
            s.pop("outcome", None)
            continue
        ref = scored[id(to_score[i])]
        i += 1
        s["p_target_mean"] = ref.get("p_target_mean")
        s["p_target_min"] = ref.get("p_target_min")
        s["outcome"] = ref.get("outcome")
        if s.get("p_target_replica") is not None and s["p_target_mean"] is not None:
            replica_diffs.append(abs(s["p_target_mean"] - s["p_target_replica"]))
    print(f"official-scored valid samples: {i}")

    # ---- position compliance per hint (valid samples whose request has a hint)
    pos_compliance = hint_compliance(samples, hints)

    # ---- per-cell report
    per_cell = {}
    for variant in sorted({s["variant"] for s in samples}):
        for temp in sorted({s["temperature"] for s in samples}):
            cell_samples = [
                s for s in samples if s["variant"] == variant and s["temperature"] == temp
            ]
            valid = [s for s in cell_samples if s["status"] == "valid"]
            n = len(cell_samples)
            entry = {
                "n": n,
                "valid_asrun": sum(1 for s in cell_samples if s["status_asrun"] == "valid"),
                "valid_d15": len(valid),
                "validity_asrun": round(
                    sum(1 for s in cell_samples if s["status_asrun"] == "valid") / n, 4
                ),
                "validity_d15": round(len(valid) / n, 4),
                "name_at_start_share_d15": round(
                    sum(1 for s in valid if opens_with_name(s["text"], s["target_span"]))
                    / len(valid),
                    4,
                )
                if valid
                else 0.0,
                "d15_flips": [
                    {
                        "request_id": s["request_id"],
                        "sample_idx": s["sample_idx"],
                        "asrun": s["status_asrun"],
                        "d15": s["status"],
                    }
                    for s in cell_samples
                    if s["status_asrun"] != s["status"]
                ],
            }
            pos = {}
            for s in valid:
                pos.setdefault(name_position_bin(s["target_span"], s["text"]), 0)
                pos[name_position_bin(s["target_span"], s["text"])] += 1
            entry["name_position_d15"] = (
                {k: round(v / len(valid), 4) for k, v in sorted(pos.items())} if valid else {}
            )
            if valid and any(s.get("p_target_mean") is not None for s in valid):
                clip = 10.0
                reward_in = []
                for s in valid:
                    p = s.get("p_target_mean")
                    if p is None:
                        continue
                    q = min(max(p, 1e-6), 1 - 1e-6)
                    reward_in.append(
                        {
                            "group_id": s["group_id"],
                            "status": "valid",
                            "p_target_mean": p,
                            "p_target_min": s.get("p_target_min"),
                            "outcome": s["outcome"],
                            "n_chars": len(s["text"]),
                            "name_position": (s["target_span"][0] + s["target_span"][1])
                            / 2
                            / max(len(s["text"]), 1),
                            "extra_person_flags": 0,
                            "r_prob": 1.0 - p,
                            "r_logit": max(-clip, min(clip, -math.log(q / (1 - q)))),
                        }
                    )
                reward = reward_metrics(reward_in, clip)
                entry["official_scorer"] = {
                    "scored": reward.get("valid_scored"),
                    "p_target_mean": reward.get("p_target_mean"),
                    "outcome_rates": reward.get("outcome_rates"),
                    "live_group_share_r_prob": reward.get("live_group_share_r_prob"),
                    "live_group_share_r_logit": reward.get("live_group_share_r_logit"),
                    "dead_group_share": reward.get("dead_group_share"),
                    "note": "official probe scorer (probe._score_with_d0), scorer verified by PR #5",
                }
            per_cell[f"{variant}|t{temp}"] = entry

    report = {
        "kind": "m07b-rescore",
        "source_samples": args.samples,
        "detector": {"path": args.detector, "sha256": args.detector_sha},
        "rule_version": "D15 (refusal: answer-start or no-content only; greeting lexicon incl. 'having')",
        "n": len(samples),
        "replica_vs_official": {
            "common_samples": len(replica_diffs),
            "max_abs_p_target_diff": max(replica_diffs) if replica_diffs else None,
            "mean_abs_p_target_diff": round(sum(replica_diffs) / len(replica_diffs), 8)
            if replica_diffs
            else None,
        },
        "per_cell": per_cell,
        "position_compliance_by_hint": pos_compliance,
    }

    out_dir = os.path.dirname(os.path.abspath(args.samples))
    with open(os.path.join(out_dir, "samples_rescored.jsonl"), "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "rescore_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nper-cell (asrun -> D15 validity):")
    for k, m in sorted(per_cell.items()):
        m = cast("dict[str, Any]", m)
        print(
            f"  {k}: {m['validity_asrun']} -> {m['validity_d15']} "
            f"(name-at-start {m['name_at_start_share_d15']})"
        )
        o = m.get("official_scorer")
        if o and o.get("p_target_mean") and o.get("outcome_rates"):
            print(
                f"      p_target p50={o['p_target_mean']['p50']:.4f} "
                f"miss={o['outcome_rates']['complete_miss']} live={o['live_group_share_r_prob']}"
            )
    print("\nposition compliance by hint:")
    for hint, cells in sorted(pos_compliance.items()):
        for cell, c in cells.items():
            print(f"  hint={hint:<9} {cell}: {c['compliant']}/{c['n']} ({c['rate']})")
    if replica_diffs:
        print(
            f"\nreplica vs official p_target: max diff {max(replica_diffs):.2e}, "
            f"mean {sum(replica_diffs) / len(replica_diffs):.2e} over {len(replica_diffs)} samples"
        )
    print(f"\nrescored samples -> {os.path.join(out_dir, 'samples_rescored.jsonl')}")
    print(f"report -> {os.path.join(out_dir, 'rescore_report.json')}")


if __name__ == "__main__":
    main()
