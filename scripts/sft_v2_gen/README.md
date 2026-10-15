# sft_v2_gen — deterministic text generator for the spark-sft-v1 dataset

Seeded pipeline (quotas -> authored batches); `scripts/validate_sft_v2.py` is
ground truth after every step. Final rows live in
`data/synthetic/spark_sft_v1/` (see `manifest.json`).

Relabeled 2026-10-15 (D16): rows are `provenance: template_paraphrase`, usable
only as the M10 template-baseline arm. `short_eval_v1.jsonl` is DEPRECATED and
moved to `data/deprecated/` (template contamination with train) — do not
regenerate or reuse it for evaluation.

Order: `gen_v2.py` (quotas must exist: `make_sft_v2_quotas.py` + gap/dev100
steps in manifest) -> `rewrite_residuals.py` (Jaccard acceptance) ->
hand-written replacements for stubborn clusters (in data, documented in
manifest) -> `scripts/finalize_sft_v2.py`. Seeds: quotas 20260920, texts
777001, dev100 20260920. Do NOT re-run casually: outputs are committed.
