"""Label-scheme ablation for limitation L1 (docs/LIMITATIONS.md): BIO vs PERSON / O.

Trains one ModernBERT-base detector from scratch on the AI4Privacy train side with either
the BIO labels the project used (O, B-PERSON, I-PERSON) or plain PERSON / O, with identical
code, seed, data order and hyperparameters (the M03 recipe: lr 5e-5, batch 16, 3 epochs,
max length 512, AdamW, linear decay). Both are evaluated the same way, with the PERSON /
not-PERSON reading (D25), on AI4Privacy dev (final-eval ids excluded) and the short_eval sets.
This is a twin comparison under this script, not a re-creation of the base detector.

Usage: PYTHONPATH=src python scripts/train_label_scheme_ablation.py --scheme io --cache ~/cache \
         --out ~/runs/ablation-io-<stamp>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, os.path.dirname(__file__))

from pii_redteam.detector import bio_labels_from_spans  # noqa: E402
from pii_redteam.detector_metrics import (  # noqa: E402
    load_negative_rows,
    load_synthetic_rows,
    span_metrics,
    strip_per_row,
)
from pii_redteam.train_detector import (  # noqa: E402
    DEV_FILE,
    TRAIN_FILE,
    _final_eval_ids,
    _load_ai4privacy,
    predict_spans,
)

MODEL_ID = "answerdotai/ModernBERT-base"
FINAL_EVAL_SHA = "513335badf5752a340698d3c7efa7029a8780c51fde2675c80ef4208ce27d39c"
SCHEMES = {
    "bio": {"O": 0, "B-PERSON": 1, "I-PERSON": 2},
    "io": {"O": 0, "PERSON": 1},
}


def token_label_ids(offsets, spans, label2id: dict[str, int]) -> list[int]:
    """BIO labels from the spans, collapsed to PERSON for the io scheme; specials get -100."""
    labels = bio_labels_from_spans(offsets, spans)
    ids = []
    for (start, end), label in zip(offsets, labels, strict=True):
        if start == end:
            ids.append(-100)
        elif "PERSON" in label2id:
            ids.append(label2id["O"] if label == "O" else label2id["PERSON"])
        else:
            ids.append(label2id[label])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=sorted(SCHEMES), required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="smoke test: first N train/dev rows")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"

    label2id = SCHEMES[args.scheme]
    excluded = _final_eval_ids({"locks": {"final_eval_sha256": FINAL_EVAL_SHA}}, "artifacts/splits")
    train_rows = _load_ai4privacy(os.path.join(args.cache, TRAIN_FILE), excluded)
    dev_rows = _load_ai4privacy(os.path.join(args.cache, DEV_FILE), excluded)
    if args.limit:
        train_rows, dev_rows = train_rows[: args.limit], dev_rows[: args.limit]
    evals = {
        "short_eval_v2": load_synthetic_rows("data/synthetic/short_eval_v2.jsonl"),
        "short_eval_deepseek_v1": load_synthetic_rows(
            "data/synthetic/short_eval_deepseek_v1.jsonl"
        ),
    }
    negatives = load_negative_rows("data/synthetic/short_eval_neg_v1.jsonl")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(label2id),
        id2label={i: label for label, i in label2id.items()},
        label2id=label2id,
    ).to(device)

    encoded = []
    for row in train_rows:
        enc = tok(
            row["text"], return_offsets_mapping=True, truncation=True, max_length=args.max_length
        )
        encoded.append(
            {
                "input_ids": enc["input_ids"],
                "labels": token_label_ids(enc["offset_mapping"], row["spans"], label2id),
            }
        )

    def collate(batch):
        width = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id or 0
        return {
            "input_ids": torch.tensor(
                [b["input_ids"] + [pad] * (width - len(b["input_ids"])) for b in batch]
            ),
            "attention_mask": torch.tensor(
                [[1] * len(b["input_ids"]) + [0] * (width - len(b["input_ids"])) for b in batch]
            ),
            "labels": torch.tensor(
                [b["labels"] + [-100] * (width - len(b["labels"])) for b in batch]
            ),
        }

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        encoded, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=generator
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, args.epochs * len(loader))

    report: dict = {
        "scheme": args.scheme,
        "config": vars(args),
        "data": {"train_rows": len(train_rows), "dev_rows": len(dev_rows)},
        "epochs": [],
    }
    best_f1 = -1.0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item():.4f}", flush=True)
        model.eval()
        with torch.no_grad():
            dev = span_metrics(
                dev_rows, predict_spans(model, tok, dev_rows, args.max_length, 32, device)
            )
            scored = {
                name: strip_per_row(
                    span_metrics(rows, predict_spans(model, tok, rows, args.max_length, 32, device))
                )
                for name, rows in evals.items()
            }
            neg_preds = predict_spans(model, tok, negatives, args.max_length, 32, device)
        entry = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 1),
            "dev": strip_per_row(dev),
            **scored,
            "short_eval_neg_v1_flagged": sum(1 for p in neg_preds if p),
        }
        report["epochs"].append(entry)
        print(
            json.dumps({k: entry[k] for k in ("epoch", "seconds")} | {"dev_f1": dev["span_f1"]}),
            flush=True,
        )
        if dev["span_f1"] > best_f1:
            best_f1 = dev["span_f1"]
            report["best_epoch"] = epoch
            model.save_pretrained(os.path.join(args.out, "best"))
            tok.save_pretrained(os.path.join(args.out, "best"))
        with open(os.path.join(args.out, "ablation_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
