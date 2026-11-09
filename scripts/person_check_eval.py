"""Prepare eval rows + judge controls for scripts/person_check_v2.py.

Converts the short-message eval sets to the sample format the person checker
reads, and appends hand-written controls. The controls exist because the judge
answered "yes" on all 673 samples of the M07b run, so its ability to say "no"
was never measured (docs/PROMPT_V2_RESULTS.md). The checker groups its report by
``variant``, which here is the dataset or control group.

Usage (the second step needs the ml group):
  PYTHONPATH=src uv run python scripts/person_check_eval.py --out /tmp/eval_samples.jsonl
  PYTHONPATH=src uv run --group ml python scripts/person_check_v2.py \
      --samples /tmp/eval_samples.jsonl --out-dir results/short_eval_person_check
"""

from __future__ import annotations

import argparse
import json

DATASETS = {
    "short_eval_v2": "data/synthetic/short_eval_v2.jsonl",
    "short_eval_deepseek_v1": "data/synthetic/short_eval_deepseek_v1.jsonl",
}

# (name, text): the name is clearly NOT a person. A working judge answers "no".
CONTROLS_NON_PERSON = (
    ("Peru", "We spent two weeks hiking in Peru last spring and the mountain views were superb."),
    ("Church", "Turn left at the old Church and the bakery is the second door on the right."),
    ("Taekwondo", "Taekwondo practice moved to the small gym, so bring your own mat on Thursday."),
    (
        "Franklin",
        "The Franklin stove in the cabin needs a new flue before the cold weather starts.",
    ),
    ("Pika", "The Pika is a small mountain mammal that gathers grass and stores it for winter."),
    ("Monterosa", "The ski lifts at Monterosa open at nine, and the valley road closes at dusk."),
    ("George", "The George is a quiet pub by the river that serves food until nine each night."),
    ("Madison", "Our flight lands in Madison at noon, so the conference shuttle should be easy."),
    ("Amazon", "The replacement cable from Amazon arrived today, but the box was badly dented."),
    ("Paris", "The Paris office closes early on Friday because of the building inspection."),
)

# (name, text): the name is clearly a person. A working judge answers "yes".
CONTROLS_PERSON = (
    ("Peru", "Ms. Peru from accounting signed the invoice and sent the copy back this morning."),
    ("Church", "Mr. Church called to say that he will miss the review because his train is late."),
    ("Franklin", "Franklin said she would finish the slides tonight and share them with us."),
    ("George", "I asked George to water my plants while I am away, and he happily agreed."),
    ("Madison", "Madison is bringing her laptop to the workshop because she will demo the app."),
    ("Keegan", "Keegan finished his shift early, so he can cover the front desk after lunch."),
    ("Rajeev", "Rajeev reviewed the contract yesterday and asked two questions about the dates."),
    ("Aneta", "Please thank Aneta for staying late, because she fixed the broken export job."),
    ("Ezekiel", "Ezekiel booked the room for ten and told everyone that he would bring coffee."),
    ("Diana-Carmen", "Diana-Carmen sent her notes to the team and asked for comments by Friday."),
)


def sample(*, variant: str, request_id: str, name: str, text: str) -> dict:
    """One record in the shape scripts/person_check_v2.py expects."""
    return {
        "variant": variant,
        "temperature": 0.0,
        "request_id": request_id,
        "sample_idx": 0,
        "status": "valid",
        "name": name,
        "text": text,
    }


def build_samples() -> list[dict]:
    samples = []
    for dataset, path in DATASETS.items():
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                samples.append(
                    sample(
                        variant=dataset,
                        request_id=row["id"],
                        name=row["request"]["entities"][0]["value"],
                        text=row["target"]["text"],
                    )
                )
    for group, controls in (
        ("control_non_person", CONTROLS_NON_PERSON),
        ("control_person", CONTROLS_PERSON),
    ):
        for i, (name, text) in enumerate(controls):
            samples.append(
                sample(variant=group, request_id=f"{group}-{i:02d}", name=name, text=text)
            )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    samples = build_samples()
    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"{len(samples)} samples -> {args.out}")


if __name__ == "__main__":
    main()
