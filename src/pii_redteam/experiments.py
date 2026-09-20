"""M01 smoke workflow: data -> generate -> validate -> reward -> evaluate -> report.

Runs on fake CPU models only. The report is stamped MOCK and must never be
presented as a research result (plan section 4: mocks excluded from science).
"""

from __future__ import annotations

import json
from pathlib import Path

from . import evaluation
from .config import config_hash, save_resolved
from .detector import FakeDetector
from .generator import FakeGenerator
from .manifest import complete_manifest, guard_resume, new_manifest, save_manifest
from .rewards import score_candidate
from .validation import validate_annotated_example

MOCK_NOTICE = "MOCK run: fake detector/generator; outputs excluded from scientific results"


def run_smoke(cfg: dict, run_dir: str | Path, examples: list[dict]) -> dict:
    """Execute the smoke workflow into an isolated run dir; return the report dict."""
    run_dir = Path(run_dir)
    chash = config_hash(cfg)
    guard_resume(
        run_dir, model_id=cfg["model_id"], config_hash=chash, resume=bool(cfg.get("resume"))
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(
        new_manifest(model_id=cfg["model_id"], config_hash=chash, seed=cfg["seed"]), run_dir
    )
    save_resolved(cfg, run_dir / "resolved_config.yaml")

    schema_errors = [
        f"{ex.get('example_id')}: {err}"
        for ex in examples
        for err in validate_annotated_example(ex)
    ]
    if schema_errors:
        complete_manifest(
            run_dir, status="failed-schema", counters={"schema_errors": len(schema_errors)}
        )
        raise ValueError("fixture schema errors: " + "; ".join(schema_errors))

    detector = FakeDetector()
    agg_tp = agg_pred = agg_gold = 0
    for ex in examples:
        m = evaluation.exact_span_prf(detector.predict_spans(ex["text"]), ex["spans"])
        agg_tp += m["tp"]
        agg_pred += len(detector.predict_spans(ex["text"]))
        agg_gold += len(ex["spans"])
    precision = agg_tp / agg_pred if agg_pred else 0.0
    recall = agg_tp / agg_gold if agg_gold else 0.0

    generator = FakeGenerator(model_id=cfg["model_id"])
    names = cfg["data"].get("names") or ["Ana Ruiz"]
    max_examples = int(cfg["data"]["max_examples"])
    budget = int(cfg["budget"]["max_candidates"])
    candidates: list[dict] = []
    idx = 0
    for round_no in range(max_examples):
        for name in names:
            if len(candidates) >= budget:
                break
            text = generator.generate(supplied_name=name)
            reward = score_candidate(text=text, supplied_name=name, detector=detector)
            candidates.append(
                {
                    "candidate_id": f"cand-{idx:03d}",
                    "generation_method": "fake-generate",
                    "generator_checkpoint": generator.model_id,
                    "serving_backend": cfg["backend"],
                    "sampling": {"temperature": 0.0},
                    "supplied_name": name,
                    "text": text,
                    "valid": reward["valid"],
                    "validity_reason": reward["reason"],
                    "reward": reward,
                    "selection": "unselected",
                    "cost": {"label": "MOCK", "value": 0.0},
                }
            )
            idx += 1
        _ = round_no

    n_valid = sum(1 for c in candidates if c["valid"])
    report = {
        "mock": True,
        "notice": MOCK_NOTICE,
        "detector_eval": {
            "model": detector.model_id,
            "tp": agg_tp,
            "precision": precision,
            "recall": recall,
            "n_examples": len(examples),
        },
        "counts": {
            "candidates": len(candidates),
            "valid": n_valid,
            "invalid": len(candidates) - n_valid,
        },
        "candidates": candidates,
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    complete_manifest(
        run_dir,
        counters={"candidates": len(candidates), "valid": n_valid, "eval_examples": len(examples)},
    )
    return report
