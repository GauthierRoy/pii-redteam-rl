# Deprecated data — do not use

## spark_sft_v1_short_eval_v1.jsonl

Moved here from `data/synthetic/spark_sft_v1/short_eval_v1.jsonl` (2026-10-15,
`fix/spark-sft-v1-relabel`).

**Status: CONTAMINATED — must never be used for evaluation.**

- Its texts come from the same seeded template bank as the training split
  (`scripts/sft_v2_gen/templates_1.py` / `templates_2.py`, seed 777001). Measured
  template overlap with `train.jsonl`: **298 of 300 rows share at least one
  name/number-normalized 8-word shingle with a train row** (example:
  `short_eval-0009` vs `train-0004/0007/0010`, same support-ticket template).
  An eval set drawn from the same template distribution as train cannot
  measure generalization; it measures template memorization.
- The generator also padded texts to word-count buckets with repeated filler
  fragments ("Agreed unanimously.", "Duly recorded.", ...), which the near-dup
  Jaccard>0.7 contract does not catch, and its synonym swaps produced
  artifacts ("speeds yet collapse").
- Rows are relabeled `provenance: template_paraphrase` like the rest of
  spark_sft_v1; row content is otherwise frozen as merged in PR #7.

A replacement short-message eval set must be authored from held-out names with
an independent method (per D13: held-out names plus a style-independent,
~50-row human-written/other-LLM slice), not from the spark_sft_v1 template bank.
