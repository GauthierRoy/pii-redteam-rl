"""Audit negative-row false positives of a trained detector checkpoint (M03.3 follow-up).

Runs the checkpoint over validation-file rows that hold no PERSON spans, decodes
predicted spans, and classifies each false positive: does the predicted string appear
in the row's own `privacy_mask` (and under which label — e.g. USERNAME/TITLE, i.e.
policy-excluded but name-like), or is it a template artifact, or an unannotated
name-like string?

Usage:
  uv run --group ml python scripts/audit_negative_fps.py \
    --checkpoint <d0-best dir> --cache-dir /tmp/pii-redteam-cache

Note: this audit took ~40 min on laptop CPU (2026-10-10). Convention (STATE.md):
run audits on Colab GPU (checkpoint + cache already on Drive) —
    python scripts/audit_negative_fps.py --checkpoint <Drive>/d0-best ...
ONNX export for local CPU is deferred.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from pii_redteam.data import QuarantinedRecord, to_canonical
from pii_redteam.detector import spans_from_bio_labels, trim_span_whitespace

MAX_LENGTH = 512


def classify(value: str, mask_labels: dict[str, str]) -> dict:
    """Classify one FP string against the row's own mask values."""
    labels = sorted({label for mask_value, label in mask_labels.items() if value in mask_value})
    if labels:
        return {"kind": "mask_value", "labels": labels}
    if re.search(r"\d", value):
        return {"kind": "digit_artifact", "labels": []}
    if re.search(r"[.|\\@/]", value):
        return {"kind": "separator_artifact", "labels": []}
    if len(value.strip()) < 2:
        return {"kind": "single_or_empty", "labels": []}
    return {"kind": "unannotated_name_like", "labels": []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", default="/tmp/pii-redteam-cache")
    parser.add_argument("--manifest", default="artifacts/splits/final_eval_manifest.json")
    parser.add_argument("--out", default="artifacts/d0_audit/negative_fp_audit.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    final_eval_ids = {r["example_id"] for r in manifest["rows"]}

    negatives = []  # (canonical, mask_values_by_text)
    with open(os.path.join(args.cache_dir, "1english_openpii_8k.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                canonical = to_canonical(json.loads(line))
            except QuarantinedRecord:
                continue
            if canonical["spans"] or canonical["example_id"] in final_eval_ids:
                continue
            mask_labels = {m["value"]: m["label"] for m in json.loads(line)["privacy_mask"]}
            negatives.append((canonical, mask_labels))
    print(f"negative dev rows: {len(negatives)}")

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint).to(device)
    model.eval()

    fps = []
    with torch.no_grad():
        for i in range(0, len(negatives), args.batch_size):
            chunk = negatives[i : i + args.batch_size]
            enc = tok(
                [c["text"] for c, _ in chunk],
                return_offsets_mapping=True,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
            )
            input_ids = torch.tensor(enc["input_ids"]).to(device)
            attn = torch.tensor(enc["attention_mask"]).to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            for j, (row, mask_labels) in enumerate(chunk):
                offsets = enc["offset_mapping"][j]
                pred_ids = logits[j].argmax(-1).tolist()
                labels = [model.config.id2label[p] for p in pred_ids]
                decoded = trim_span_whitespace(row["text"], spans_from_bio_labels(offsets, labels))
                for sp in decoded:
                    value = row["text"][sp["start"] : sp["end"]]
                    fps.append(
                        {
                            "example_id": row["example_id"],
                            "value": value,
                            **classify(value, mask_labels),
                        }
                    )
            if (i // args.batch_size) % 25 == 0:
                print(f"  batch {i // args.batch_size}/{len(negatives) // args.batch_size}")

    summary: dict[str, int] = {}
    for fp in fps:
        key = fp["kind"] + (f"({','.join(fp['labels'])})" if fp["labels"] else "")
        summary[key] = summary.get(key, 0) + 1

    report = {
        "checkpoint": args.checkpoint,
        "negative_rows": len(negatives),
        "false_positives": len(fps),
        "rows_with_fp": len({fp["example_id"] for fp in fps}),
        "summary": dict(sorted(summary.items(), key=lambda kv: -kv[1])),
        "examples": {k: [fp["value"] for fp in fps if fp["kind"] + (f"({','.join(fp['labels'])})" if fp["labels"] else "") == k][:12] for k in summary},
        "detail": fps,
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("negative_rows", "false_positives", "rows_with_fp", "summary")}, indent=2))
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
