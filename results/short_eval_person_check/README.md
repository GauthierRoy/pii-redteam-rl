# Person check of the short-message eval sets (2026-11-08)

Run on local CPU with `scripts/person_check_eval.py` and `scripts/person_check_v2.py`, at
commit `9b3b317`. The judge was Qwen3.5-0.8B. The `checkers` field in
`person_check_report.json` says 2B because that text is hardcoded in the script.

## Do not treat the person-sense judge as a validity gate

| Controls | Judge answered "yes, a person" | Correct |
|---|---|---|
| 10 names clearly used as a person | 10/10 | 10/10 |
| 10 names clearly not used as a person | 8/10 | 2/10 |

It said yes to "Taekwondo practice moved to the small gym" and "The Paris office closes
early". Its 0 rejections across the 200 eval rows are therefore not evidence that the labels
are right. This also affects the 0/673 person-sense result in `docs/PROMPT_V2_RESULTS.md`
and the R08 gate, which rely on the same judge. The controls are few and hand-written, so
this shows weak discrimination rather than a precise error rate.

## Manual review of all 200 rows

Each name was read in its surrounding text.

- **Likely mislabelled:** `short-eval-v2-0016`, "After the office manager explained the
  clash to Taekwondo, both clubs agreed…". It reads as a club, but it is labelled PERSON.
  PR #18 relabels it to the held-out name `Petrea`; the files in this directory record the
  run on the original row.
- **Unusual but acceptable:** an honorific on a name that looks like a given name, such as
  "Ms. Lívio" and "Mr. Aiden". These labels are still person mentions.

This covered the person sense of the target name only. Extra-person coverage comes from
spaCy and the judge:

- spaCy flagged 10 rows, all false positives: "Cheers" (6), "metadata", "Mill Lane", "Keys",
  "Volunteer".
- The judge's 13 unclear answers come from its yes/no parsing and do not point to a real
  mention. Do not use `accepted_for_d1` from this report as an eval-row filter.
