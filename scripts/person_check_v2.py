"""Person checks over saved comparison samples (M07b.2 D1 training-data gate).

Usage (needs the ml group):
  PYTHONPATH=src uv run --group ml python scripts/person_check_v2.py \
      --samples results/m07b_prompt_v2/m07b-20261012-224819/samples_rescored.jsonl \
      --out-dir results/m07b_prompt_v2/m07b-20261012-224819

For every D15-valid sample:
  1. spaCy NER (`en_core_web_sm`): extra PERSON entities besides the requested name.
  2. LLM judge (base Qwen3.5-2B, non-thinking, greedy): extra-person yes/no +
     person-sense yes/no.

Row rejected for D1 training data if any extra person (spaCy OR LLM yes) or the
person-sense answer is not "yes". Writes samples_person_checked.jsonl,
person_check_report.json, and audit_sample_50.jsonl (stratified 25 flagged /
25 clean, for the manual audit).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pii_redteam.person_check import (
    EXTRA_PERSON_PROMPT,
    PERSON_SENSE_PROMPT,
    parse_yes_no,
    spacy_extra_persons,
)


def _chat_text(tok, prompt: str) -> str:
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )


def judge_batch(model, tok, prompts: list[str], device, batch_size: int = 64) -> list[bool | None]:
    """Greedy yes/no answers for a list of full judge prompts."""
    tok.padding_side = "left"
    answers: list[bool | None] = []
    t_start = time.time()
    for i in range(0, len(prompts), batch_size):
        t0 = time.time()
        chunk = prompts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=2, do_sample=False)
        n_prompt = enc["input_ids"].shape[1]
        for j in range(len(chunk)):
            answers.append(parse_yes_no(tok.decode(gen[j][n_prompt:], skip_special_tokens=True)))
        print(
            f"  judged {i + len(chunk)}/{len(prompts)} ({time.time() - t0:.0f}s batch, "
            f"{time.time() - t_start:.0f}s total)",
            flush=True,
        )
    return answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="samples_rescored.jsonl (D15 labels)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--skip-llm", action="store_true", help="spaCy-only pass")
    parser.add_argument(
        "--limit", type=int, default=None, help="debug: check only the first N valid samples"
    )
    args = parser.parse_args()

    with open(args.samples, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    valid = [s for s in samples if s["status"] == "valid"]
    if args.limit:
        valid = valid[: args.limit]
    print(f"samples: {len(samples)} total, {len(valid)} D15-valid to check")

    # ---- spaCy NER pass
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    t0 = time.time()
    for s in valid:
        s["spacy_extra_persons"] = spacy_extra_persons(nlp(s["text"]), s["name"])
    print(
        f"spaCy pass done in {time.time() - t0:.1f} s "
        f"({sum(1 for s in valid if s['spacy_extra_persons'])} rows with extras)"
    )

    # ---- LLM judge pass
    if not args.skip_llm:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cpu" else torch.float16
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token_id is None:  # ty: ignore[unresolved-attribute]
            tok.pad_token = tok.eos_token  # ty: ignore[invalid-assignment,unresolved-attribute]
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)  # ty: ignore[invalid-argument-type]  # stub union
        model.eval()
        print(f"LLM judge: {args.model} on {device} (greedy, non-thinking)")

        t0 = time.time()
        extra_prompts = [
            _chat_text(tok, EXTRA_PERSON_PROMPT.format(text=s["text"], name=s["name"]))
            for s in valid
        ]
        sense_prompts = [
            _chat_text(tok, PERSON_SENSE_PROMPT.format(text=s["text"], name=s["name"]))
            for s in valid
        ]
        with torch.no_grad():
            print("judging extra-person question ...")
            extra_answers = judge_batch(model, tok, extra_prompts, device)
            print("judging person-sense question ...")
            sense_answers = judge_batch(model, tok, sense_prompts, device)
        print(f"LLM judge done in {(time.time() - t0) / 60:.1f} min")
        for s, e, p in zip(valid, extra_answers, sense_answers, strict=True):
            s["llm_extra_person"] = e
            s["llm_person_sense"] = p
    else:
        for s in valid:
            s["llm_extra_person"] = None
            s["llm_person_sense"] = None

    # ---- verdicts (conservative: an unclear judge answer rejects the row —
    # manual audit of run m07b-20261012-224819 caught a leaked few-shot name
    # exactly in the unclear bucket)
    for s in valid:
        s["reject_extra_person"] = (
            bool(s["spacy_extra_persons"]) or s["llm_extra_person"] is not False
        )
        s["reject_extra_person_lenient"] = (
            bool(s["spacy_extra_persons"]) or s["llm_extra_person"] is True
        )
        s["reject_person_sense"] = s["llm_person_sense"] is not True
        s["accepted_for_d1"] = not (s["reject_extra_person"] or s["reject_person_sense"])

    # ---- report
    def rate(rows: list[dict], key: str) -> float:
        return round(sum(1 for r in rows if r[key]) / len(rows), 4) if rows else 0.0

    per_cell = {}
    for variant in sorted({s["variant"] for s in valid}):
        for temp in sorted({s["temperature"] for s in valid}):
            rows = [s for s in valid if s["variant"] == variant and s["temperature"] == temp]
            if not rows:
                continue
            per_cell[f"{variant}|t{temp}"] = {
                "n_valid": len(rows),
                "spacy_extra_rate": round(
                    sum(1 for r in rows if r["spacy_extra_persons"]) / len(rows), 4
                ),
                "llm_extra_yes_rate": round(
                    sum(1 for r in rows if r["llm_extra_person"] is True) / len(rows), 4
                ),
                "llm_extra_unclear": sum(1 for r in rows if r["llm_extra_person"] is None),
                "llm_sense_no_rate": round(
                    sum(1 for r in rows if r["llm_person_sense"] is False) / len(rows), 4
                ),
                "llm_sense_unclear": sum(1 for r in rows if r["llm_person_sense"] is None),
                "reject_extra_person_rate": rate(rows, "reject_extra_person"),
                "reject_extra_person_lenient_rate": rate(rows, "reject_extra_person_lenient"),
                "reject_person_sense_rate": rate(rows, "reject_person_sense"),
                "accepted_for_d1_rate": rate(rows, "accepted_for_d1"),
                "accepted_for_d1_n": sum(1 for r in rows if r["accepted_for_d1"]),
            }

    report = {
        "kind": "m07b-person-check",
        "samples": args.samples,
        "n_valid_checked": len(valid),
        "checkers": {
            "extra_person": "spaCy en_core_web_sm NER OR Qwen3.5-2B LLM judge (independent of D0)",
            "person_sense": "Qwen3.5-2B LLM judge",
        },
        "per_cell": per_cell,
        "overall": {
            "reject_extra_person_rate": rate(valid, "reject_extra_person"),
            "reject_extra_person_lenient_rate": rate(valid, "reject_extra_person_lenient"),
            "reject_person_sense_rate": rate(valid, "reject_person_sense"),
            "accepted_for_d1_rate": rate(valid, "accepted_for_d1"),
            "accepted_for_d1_n": sum(1 for r in valid if r["accepted_for_d1"]),
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(
        os.path.join(args.out_dir, "samples_person_checked.jsonl"), "w", encoding="utf-8"
    ) as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "person_check_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---- stratified 50-row manual-audit sample (25 flagged, 25 clean)
    rng = random.Random(args.seed)
    flagged = [s for s in valid if not s["accepted_for_d1"]]
    clean = [s for s in valid if s["accepted_for_d1"]]
    audit = rng.sample(flagged, min(25, len(flagged))) + rng.sample(clean, min(25, len(clean)))
    rng.shuffle(audit)
    with open(os.path.join(args.out_dir, "audit_sample_50.jsonl"), "w", encoding="utf-8") as f:
        for s in audit:
            f.write(
                json.dumps(
                    {
                        "request_id": s["request_id"],
                        "variant": s["variant"],
                        "temperature": s["temperature"],
                        "sample_idx": s["sample_idx"],
                        "name": s["name"],
                        "text": s["text"],
                        "spacy_extra_persons": s["spacy_extra_persons"],
                        "llm_extra_person": s["llm_extra_person"],
                        "llm_person_sense": s["llm_person_sense"],
                        "verdict": "rejected" if not s["accepted_for_d1"] else "accepted",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("\nper-cell person-check rates:")
    for k, m in sorted(per_cell.items()):
        print(
            f"  {k}: n={m['n_valid']} spacy_extras={m['spacy_extra_rate']} "
            f"llm_extra_yes={m['llm_extra_yes_rate']} (unclear {m['llm_extra_unclear']}) "
            f"sense_no={m['llm_sense_no_rate']} (unclear {m['llm_sense_unclear']}) "
            f"accepted={m['accepted_for_d1_n']}/{m['n_valid']} ({m['accepted_for_d1_rate']})"
        )
    print(
        f"\noverall: accepted_for_d1 {report['overall']['accepted_for_d1_n']}/{len(valid)} "
        f"({report['overall']['accepted_for_d1_rate']})"
    )
    print(f"outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
