"""Re-score detectors with the PERSON / not-PERSON reading (D25, docs/LIMITATIONS.md L1).

Every detector is scored on the same sets as the M06/M10 comparison, through
`train_detector.predict_spans`, which now reads a token as PERSON when 1 - P(O) >= 0.5 and
merges consecutive PERSON tokens. Dev rows exclude the final-eval ids (locked by sha).

Usage (on the machine that holds the detectors and the AI4Privacy cache):
  PYTHONPATH=src python scripts/rescore_person_reading.py --cache ~/cache \
    --detector target=PATH --detector rl=PATH ... \
    --letters A.jsonl B.jsonl --base-texts C.jsonl --extra logodds=D.jsonl --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from score_rl_letters import held_out_rows, name_free_rows  # noqa: E402

from pii_redteam.detector_metrics import load_synthetic_rows, span_metrics  # noqa: E402
from pii_redteam.prompt_compare import _device_dtype  # noqa: E402
from pii_redteam.train_detector import (  # noqa: E402
    DEV_FILE,
    _final_eval_ids,
    _load_ai4privacy,
    _load_detector,
    predict_spans,
)

FINAL_EVAL_SHA = "513335badf5752a340698d3c7efa7029a8780c51fde2675c80ef4208ce27d39c"
SHORT_EVAL = {
    "short_eval_v2": "data/synthetic/short_eval_v2.jsonl",
    "short_eval_deepseek_v1": "data/synthetic/short_eval_deepseek_v1.jsonl",
}


def summary(metrics: dict) -> dict:
    keys = ("rows", "span_f1", "span_recall", "span_precision", "complete_miss_rate")
    out = {k: metrics.get(k) for k in keys}
    out["complete_miss"] = metrics.get("outcomes", {}).get("complete_miss")
    if metrics.get("neg_rows"):
        out["neg_rows"] = metrics["neg_rows"]
        out["neg_rows_with_pred"] = metrics["neg_rows_with_pred"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--detector", action="append", required=True, help="label=path")
    parser.add_argument("--letters", nargs="+", required=True)
    parser.add_argument("--base-texts", nargs="+", required=True)
    parser.add_argument("--extra", action="append", default=[], help="label=samples.jsonl")
    parser.add_argument("--neg", default="data/synthetic/short_eval_neg_v1.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device, _, _ = _device_dtype()
    from transformers import AutoTokenizer

    excluded = _final_eval_ids({"locks": {"final_eval_sha256": FINAL_EVAL_SHA}}, "artifacts/splits")
    letters = held_out_rows(args.letters)
    sets: dict[str, list[dict]] = {
        "ai4privacy_dev": _load_ai4privacy(os.path.join(args.cache, DEV_FILE), excluded),
        **{name: load_synthetic_rows(path) for name, path in SHORT_EVAL.items()},
        "held_out_dear_letters": [r for r in letters if r["dear"]],
        "held_out_other_rl": [r for r in letters if not r["dear"]],
        "held_out_base_texts": held_out_rows(args.base_texts),
    }
    for item in args.extra:
        label, path = item.split("=", 1)
        sets[label] = held_out_rows([path])
    negatives = name_free_rows(args.neg)

    result: dict[str, dict] = {"reading": "PERSON / not-PERSON (D25)", "detectors": {}}
    for item in args.detector:
        label, path = item.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path)
        model = _load_detector(path, device)
        scored: dict[str, dict] = {"path": path}
        for name, rows in sets.items():
            scored[name] = summary(
                span_metrics(rows, predict_spans(model, tok, rows, 512, 32, device))
            )
        for name, rows in negatives.items():
            preds = predict_spans(model, tok, rows, 512, 32, device)
            scored[name] = {"rows": len(rows), "rows_with_pred": sum(1 for p in preds if p)}
        result["detectors"][label] = scored
        print(label, json.dumps({k: v for k, v in scored.items() if k != "path"}))
        del model
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
