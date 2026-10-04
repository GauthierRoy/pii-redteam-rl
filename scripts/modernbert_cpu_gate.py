"""ModernBERT CPU gate (M03.1, laptop-verifiable part).

Loads `answerdotai/ModernBERT-base` as a token-classification model with its own
tokenizer, checks the repo's BIO alignment round-trips on real train-side rows
(final-eval rows excluded), and runs one real optimizer step. No scientific result.

Requires the optional ML group: `uv run --group ml python scripts/modernbert_cpu_gate.py`
Dataset cache comes from `scripts/carve_splits.py --cache-dir ...` (default /tmp).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from pii_redteam.data import QuarantinedRecord, to_canonical
from pii_redteam.detector import bio_labels_from_spans, spans_from_bio_labels, trim_span_whitespace

MB_ID = "answerdotai/ModernBERT-base"
LABEL2ID = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}


def load_train_rows(n: int, cache_dir: str, final_eval_ids: set[str]) -> list[dict]:
    rows = []
    path = os.path.join(cache_dir, "1english_openpii_30k.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(rows) >= n:
                break
            try:
                canonical = to_canonical(json.loads(line))
            except QuarantinedRecord:
                continue
            if canonical["example_id"] in final_eval_ids or not canonical["spans"]:
                continue
            rows.append(canonical)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MB_ID)
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--cache-dir", default="/tmp/pii-redteam-cache")
    parser.add_argument("--manifest", default="artifacts/splits/final_eval_manifest.json")
    parser.add_argument("--out", default="artifacts/cpu_gate/modernbert_cpu_report.json")
    args = parser.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    final_eval_ids = {r["example_id"] for r in manifest["rows"]}
    rows = load_train_rows(args.rows, args.cache_dir, final_eval_ids)
    print(f"train-side rows loaded: {len(rows)}")

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_id,
        num_labels=len(LABEL2ID),
        id2label={i: label for label, i in LABEL2ID.items()},
        label2id=LABEL2ID,
    )
    model.eval()
    print(f"model loaded: {args.model_id} (CPU)")

    exact = mismatched = truncated = 0
    mismatches = []
    t0 = time.time()
    for row in rows:
        enc = tok(  # ty: ignore[call-non-callable]  # transformers 5.x stub union includes None
            row["text"], return_offsets_mapping=True, truncation=True, max_length=512
        )
        if len(enc["input_ids"]) >= 512:
            truncated += 1
            continue
        offsets = enc["offset_mapping"]
        labels = bio_labels_from_spans(offsets, row["spans"])
        recovered = trim_span_whitespace(row["text"], spans_from_bio_labels(offsets, labels))
        if recovered == row["spans"]:
            exact += 1
        else:
            mismatched += 1
            mismatches.append(
                {"example_id": row["example_id"], "orig": row["spans"], "got": recovered}
            )
    roundtrip_s = time.time() - t0
    print(
        f"round-trip: exact={exact} mismatched={mismatched} skipped={truncated} ({roundtrip_s:.1f}s)"
    )
    if mismatches:
        print(json.dumps(mismatches[:3], indent=2, ensure_ascii=False))

    model.train()
    batch = rows[:4]
    enc = tok(  # ty: ignore[call-non-callable]  # transformers 5.x stub union includes None
        [r["text"] for r in batch],
        return_offsets_mapping=True,
        truncation=True,
        max_length=128,
        padding=True,
    )
    labels = []
    for offsets, row in zip(enc["offset_mapping"], batch, strict=True):
        labels.append(
            [
                -100 if (s, e) == (0, 0) else LABEL2ID[lab]
                for (s, e), lab in zip(
                    offsets, bio_labels_from_spans(offsets, row["spans"]), strict=True
                )
            ]
        )
    input_ids = torch.tensor(enc["input_ids"])
    attention = torch.tensor(enc["attention_mask"])
    labels_t = torch.tensor(labels)
    t0 = time.time()
    out = model(input_ids=input_ids, attention_mask=attention, labels=labels_t)
    out.loss.backward()
    grad_sum = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    torch.optim.AdamW(model.parameters(), lr=5e-5).step()
    step_s = time.time() - t0
    rss_gb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    print(
        f"train step: loss={out.loss.item():.4f} grad_sum={grad_sum:.3e} step={step_s:.2f}s peak_rss={rss_gb} GB"
    )

    report = {
        "mode": "cpu",
        "model_id": args.model_id,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "rows": len(rows),
        "roundtrip_exact": exact,
        "roundtrip_mismatched": mismatched,
        "roundtrip_seconds": round(roundtrip_s, 2),
        "train_step_loss": out.loss.item(),
        "grad_sum": grad_sum,
        "train_step_seconds": round(step_s, 2),
        "peak_rss_gb": rss_gb,
        "mismatches": mismatches[:5],
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"report: {args.out}")
    if mismatched or not (torch.isfinite(out.loss) and grad_sum > 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
