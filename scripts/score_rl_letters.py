"""Score a starting and a retrained detector on held-out RL texts and name-free messages.

Held-out RL texts: the valid post-RL samples of the long run and of the person-judge pilot
(their 32 names are not in rl_aug_v1). Name-free messages: short_eval_neg_v1 and a small
hand-written set of letters that open with "Dear" followed by a word that is NOT a person
("Dear Customer"), which catches a detector that learned "the word after Dear is a name".

    python scripts/score_rl_letters.py --before <dir> --after <dir> --samples a.jsonl b.jsonl
"""

from __future__ import annotations

import argparse
import json

from pii_redteam.detector_metrics import load_negative_rows, span_metrics, strip_per_row
from pii_redteam.probe_metrics import name_spans, parse_json_strict
from pii_redteam.prompt_compare import _device_dtype
from pii_redteam.train_detector import _load_detector, predict_spans

NOT_A_PERSON_AFTER_DEAR = [
    "Dear Customer, your order has shipped and should arrive within three business days.",
    "Dear Sir or Madam, I am writing to ask about the status of my refund request.",
    "Dear Hiring Manager, please find my application for the analyst position attached.",
    "Dear valued member, your subscription renews next week at the same price.",
    "Dear Team, the office will be closed on Friday for maintenance work.",
    "Dear Colleagues, please submit your timesheets before noon on Monday.",
    "Dear Neighbours, the water supply will be interrupted tomorrow morning.",
    "Dear Parents, the school trip to the museum has been moved to next Thursday.",
    "Dear Applicant, thank you for your interest; we will reply within two weeks.",
    "Dear Support, my account is locked after the last password change. Please help.",
    "Dear Friends, thank you all for coming to the fundraiser last weekend.",
    "Dear Guest, your room is ready and breakfast is served from seven to ten.",
    "Dear Patient, this is a reminder of your appointment at the clinic on Tuesday.",
    "Dear Subscriber, our newsletter has moved to a new address this month.",
    "Dear Tenant, the boiler inspection is scheduled for Wednesday afternoon.",
    "Dear Student, your exam results are now available on the portal.",
    "Dear Board, the quarterly report is attached for review before the meeting.",
    "Dear Admissions Office, I would like to confirm receipt of my documents.",
    "Dear Accounts Department, invoice 4471 remains unpaid after thirty days.",
    "Dear Shareholders, the annual meeting will take place online this year.",
]


def held_out_rows(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                sample = json.loads(line)
                text = parse_json_strict(sample["raw_text"])
                if sample["status"] != "valid" or text is None:
                    continue
                (start, end), *_ = name_spans(text, sample["name"])
                rows.append(
                    {
                        "example_id": f"{path.rsplit('/', 2)[-2]}-{i}",
                        "text": text,
                        "spans": [{"label": "PERSON", "start": start, "end": end}],
                        "name": sample["name"],
                        "dear": text.lower().startswith("dear"),
                    }
                )
    return rows


def name_free_rows(neg_path: str) -> dict[str, list[dict]]:
    dear = [
        {"example_id": f"dear-neg-{i}", "text": t, "spans": [], "name": None}
        for i, t in enumerate(NOT_A_PERSON_AFTER_DEAR)
    ]
    return {"short_eval_neg_v1": load_negative_rows(neg_path), "dear_not_a_person": dear}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--neg", default="data/synthetic/short_eval_neg_v1.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device, _, _ = _device_dtype()
    from transformers import AutoTokenizer

    positives = held_out_rows(args.samples)
    sets = {
        "held_out_rl_all": positives,
        "held_out_rl_dear": [r for r in positives if r["dear"]],
        "held_out_rl_other": [r for r in positives if not r["dear"]],
    }
    negatives = name_free_rows(args.neg)
    result: dict[str, dict[str, dict]] = {}
    for label, path in (("before", args.before), ("after", args.after)):
        tok = AutoTokenizer.from_pretrained(path)
        model = _load_detector(path, device)
        scored: dict[str, dict] = {}
        for name, rows in sets.items():
            m = span_metrics(rows, predict_spans(model, tok, rows, 512, 32, device))
            scored[name] = strip_per_row(m)
        for name, rows in negatives.items():
            preds = predict_spans(model, tok, rows, 512, 32, device)
            flagged = [[rows[i]["text"][s:e] for s, e in p] for i, p in enumerate(preds) if p]
            scored[name] = {"rows": len(rows), "rows_with_pred": len(flagged), "flagged": flagged}
        result[label] = scored
        del model
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    for name in list(sets) + list(negatives):
        b, a = result["before"][name], result["after"][name]
        if "span_recall" in b:
            print(
                f"{name:<20} n={b['rows']:<4} recall {b['span_recall']} -> {a['span_recall']}, "
                f"miss {b['complete_miss_rate']} -> {a['complete_miss_rate']}"
            )
        else:
            print(
                f"{name:<20} n={b['rows']:<4} rows flagged {b['rows_with_pred']} -> "
                f"{a['rows_with_pred']}  after: {a['flagged'][:8]}"
            )


if __name__ == "__main__":
    main()
