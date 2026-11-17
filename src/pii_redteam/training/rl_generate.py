"""Augmentation texts: sample an attacker on fresh names and keep the usable texts.

The attacker is a trained RL adapter, or the base model when `attacker.adapter_path` is null
(the M06 plain-generation and generate-and-filter baselines).

A text is kept when it passes the validity rules and the person judge (both through the RL
reward), the audit found no other person's name in it, and it is not an exact repeat. A text
with another person's name is rejected because that name would be left unlabelled in detector
training data. The audit's accuracy on that question has not been measured.

Requests use names that the RL run never saw (neither its training nor its held-out requests).
"""

from __future__ import annotations

import datetime
import json
import os
from collections import Counter

from ..probe_metrics import name_spans, parse_json_strict
from ..run_io import write_jsonl
from .opening_share import opening_word
from .rl_bank import build_rl_requests

MISS_BELOW = 0.5
ACCEPTED = "accepted"


def with_acceptance(samples: list[dict]) -> list[dict]:
    """Copies of `samples` with `text` and `decision`: ACCEPTED or a "rejected:<reason>"."""
    seen: set[str] = set()
    decided = []
    for sample in samples:
        text = parse_json_strict(sample["raw_text"])
        if sample["status"] != "valid" or text is None:
            decision = f"rejected:{sample['status']}"
        elif sample.get("other_person") is None:
            decision = "rejected:audit_missing"
        elif sample["other_person"]:
            decision = "rejected:other_person"
        elif text in seen:
            decision = "rejected:duplicate"
        else:
            decision = ACCEPTED
            seen.add(text)
        decided.append({**sample, "text": text, "decision": decision})
    return decided


def accepted_rows(decided: list[dict]) -> list[dict]:
    """Training-ready rows: the text, the requested name and its character span."""
    rows = []
    for sample in decided:
        if sample["decision"] != ACCEPTED:
            continue
        (start, end), *_ = name_spans(sample["text"], sample["name"])
        rows.append(
            {
                "request_id": sample["request_id"],
                "sample_idx": sample.get("sample_idx", 0),
                "context_type": sample["context_type"],
                "name": sample["name"],
                "text": sample["text"],
                "name_span": [start, end],
                "p_target_mean": sample["p_target_mean"],
                "natural": sample.get("natural"),
                "fits_format": sample.get("fits_format"),
            }
        )
    return rows


def generation_counts(decided: list[dict]) -> dict:
    kept = [s for s in decided if s["decision"] == ACCEPTED]
    openings = Counter(opening_word(s["text"], s["name"]) for s in kept)
    misses_by_context = Counter(s["context_type"] for s in kept if s["p_target_mean"] < MISS_BELOW)
    return {
        "generated": len(decided),
        "decisions": dict(Counter(s["decision"] for s in decided)),
        "accepted": len(kept),
        "accepted_names": len({s["name"] for s in kept}),
        "accepted_detector_misses": sum(misses_by_context.values()),
        "accepted_misses_by_context": dict(misses_by_context),
        "accepted_by_context": dict(Counter(s["context_type"] for s in kept)),
        "accepted_top_openings": dict(openings.most_common(8)),
        "accepted_mean_words": (
            round(sum(len(s["text"].split()) for s in kept) / len(kept), 1) if kept else None
        ),
    }


def generation_requests(pool: frozenset[str], excluded: frozenset[str], bank: dict) -> list[dict]:
    """`bank.n` requests on names outside `excluded` and outside the RL run's own bank."""
    rl_bank = bank["exclude_rl_bank"]
    rl_requests = build_rl_requests(pool, n=rl_bank["n"], seed=rl_bank["seed"], exclude=excluded)
    rl_names = {e["value"] for r in rl_requests for e in r["task"]["entities"]}
    return build_rl_requests(
        pool,
        n=bank["n"],
        seed=bank["seed"],
        exclude=excluded | frozenset(rl_names),
        prefix=bank["prefix"],
    )


def run_rl_generate(config: dict, repo_dir: str) -> dict:
    import torch
    from peft import PeftModel

    from ..prompt_compare import _device_dtype
    from . import rl_smoke

    started = datetime.datetime.now(datetime.UTC)
    run_id = started.strftime(f"{config['run_id_prefix']}-%Y%m%d-%H%M%S")
    work_dir = os.path.join(rl_smoke.resolve(config["work_dir"], repo_dir), run_id)
    os.makedirs(work_dir, exist_ok=True)
    device, gpu, dtype = _device_dtype()

    pool, pool_provenance, excluded = rl_smoke.load_name_pool(config, repo_dir)
    requests = generation_requests(pool, excluded, config["bank"])
    write_jsonl(os.path.join(work_dir, "requests.jsonl"), requests)

    adapter_path = config["attacker"].get("adapter_path")  # None: the base attacker (M06)
    tok, model = rl_smoke.load_attacker(config, dtype)
    model = model.to(device)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.eval()
    scorer = rl_smoke.load_scorer(config, device)
    judge, audit = rl_smoke.person_judge(config), rl_smoke.text_audit(config)
    if judge is None or audit is None:
        raise SystemExit("generation needs reward.person_judge and reward.audit")
    torch.manual_seed(config["seed"])
    samples, _ = rl_smoke.sample_and_score(
        tok, model, requests, config, scorer, judge, audit, device, "generate"
    )
    decided = with_acceptance(samples)
    write_jsonl(os.path.join(work_dir, "samples.jsonl"), decided)
    write_jsonl(os.path.join(work_dir, "accepted.jsonl"), accepted_rows(decided))
    report = {
        "kind": "rl-generate",
        "run_id": run_id,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "config": config,
        "environment": {"gpu": gpu, "dtype": str(dtype), "torch": torch.__version__},
        "bank": {"pool": pool_provenance, "excluded_names": len(excluded)},
        "adapter_sha256": (
            rl_smoke.file_sha256(os.path.join(adapter_path, "adapter_model.safetensors"))
            if adapter_path
            else None
        ),
        "counts": generation_counts(decided),
    }
    rl_smoke.save_report(report, os.path.join(work_dir, "rl_generate_report.json"))
    print("GENERATION:", json.dumps(report["counts"], indent=2))
    return report
