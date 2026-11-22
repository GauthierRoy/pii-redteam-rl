"""Score fixed (name, text) cases with several detectors, PERSON reading (D25), on CPU.

Each case is a JSON line {"case": label, "name": str, "text": str}; the name must occur in the
text. The score is the mean PERSON probability (1 - P(O)) over the tokens of the first
occurrence of the name, as in the RL reward. A miss is a score below 0.5.

Usage: PYTHONPATH=src CUDA_VISIBLE_DEVICES= python scripts/probe_name_swap.py CASES.jsonl \
         --detector target=PATH --detector rl=PATH --out OUT.json
"""

from __future__ import annotations

import argparse
import collections
import json

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def name_score(model, tok, text: str, name: str) -> float:
    start = text.index(name)
    end = start + len(name)
    enc = tok(
        text, return_offsets_mapping=True, truncation=True, max_length=512, return_tensors="pt"
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits[0].float(), -1)
    o_id = model.config.label2id["O"]
    p_person = (1.0 - probs[:, o_id]).tolist()
    picked = [
        p for (s, e), p in zip(offsets, p_person, strict=True) if s < end and e > start and e > s
    ]
    return sum(picked) / len(picked)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases")
    parser.add_argument("--detector", action="append", required=True, help="label=path")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cases = [json.loads(line) for line in open(args.cases, encoding="utf-8")]
    result: dict = {"reading": "PERSON / not-PERSON (D25), mean over name tokens", "detectors": {}}
    for item in args.detector:
        label, path = item.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForTokenClassification.from_pretrained(path).eval()
        scores = [name_score(model, tok, c["text"], c["name"]) for c in cases]
        by_case: dict[str, list[float]] = collections.defaultdict(list)
        for case, score in zip(cases, scores, strict=True):
            by_case[case["case"]].append(score)
        summary = {
            k: {"n": len(v), "misses": sum(s < 0.5 for s in v), "mean": round(sum(v) / len(v), 3)}
            for k, v in by_case.items()
        }
        result["detectors"][label] = {"path": path, "by_case": summary, "scores": scores}
        print(label, json.dumps(summary))
    result["cases"] = cases
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
