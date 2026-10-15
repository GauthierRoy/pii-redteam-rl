"""Flag extra person names with the frozen D0 detector (flag only, never rejects).

Reads rows JSONL (id + target.text + target.span), runs the frozen D0
checkpoint in eval mode on CPU, and writes {row_id: n_extra_person_spans}
where "extra" means a predicted PERSON span that does not overlap the target
span. Also asserts the D0 model lock before scoring.

Run inside an ephemeral ML env (repo convention, torch CPU), e.g.::

    uv run --no-project --with torch --with transformers \\
      --default-index https://download.pytorch.org/whl/cpu \\
      --index https://pypi.org/simple \\
      python scripts/flag_extra_names_d0.py --rows ... --out ...

This script only *calls* frozen alignment helpers; it modifies no repo code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

D0_LOCK_SHA256 = "0314ebe36d7f4dc6d784f0e139ff5b9d32633aef1a6f88e764a610a8d8ec43cb"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True, help="JSONL with id + target.text/span")
    parser.add_argument("--out", required=True, help="flags JSON destination")
    parser.add_argument("--d0", default="/tmp/d0zip/pii-redteam-rl/d0-20261004-142714/d0-best")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from pii_redteam.detector import spans_from_bio_labels, trim_span_whitespace

    weights = os.path.join(args.d0, "model.safetensors")
    actual = hashlib.sha256(open(weights, "rb").read()).hexdigest()
    if actual != D0_LOCK_SHA256:
        raise SystemExit(f"D0 lock mismatch: {actual} != {D0_LOCK_SHA256} - refusing to score.")
    print(f"D0 lock OK ({actual[:12]}...), loading on CPU ...")

    tok = AutoTokenizer.from_pretrained(args.d0)
    model = AutoModelForTokenClassification.from_pretrained(args.d0)
    model.eval()
    id2label = model.config.id2label

    rows = []
    for line in open(args.rows, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows.append((r["id"], r["target"]["text"], tuple(r["target"]["span"])))

    flags: dict[str, int] = {}
    with torch.no_grad():
        for i in range(0, len(rows), args.batch):
            chunk = rows[i : i + args.batch]
            enc = tok(
                [t for _, t, _ in chunk],
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                padding=True,
            )
            input_ids = torch.tensor(enc["input_ids"])
            attn = torch.tensor(enc["attention_mask"])
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            for j, (rid, text, target) in enumerate(chunk):
                offsets = enc["offset_mapping"][j]
                labels = [id2label[p] for p in logits[j].argmax(-1).tolist()]
                pred = [
                    (sp["start"], sp["end"])
                    for sp in trim_span_whitespace(text, spans_from_bio_labels(offsets, labels))
                ]
                ts, te = target
                extra = sum(1 for ps, pe in pred if not (ps < te and pe > ts))
                flags[rid] = extra
            if (i // args.batch) % 5 == 0:
                print(f"  scored {min(i + args.batch, len(rows))}/{len(rows)}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(flags, f, indent=2)
    n_flagged = sum(1 for v in flags.values() if v > 0)
    print(f"flagged {n_flagged}/{len(flags)} rows -> {args.out}")


if __name__ == "__main__":
    main()
