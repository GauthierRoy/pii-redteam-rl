# sft_v2_gen — deterministic text generator for the spark-sft-v1 dataset

Seeded pipeline (quotas -> authored batches); `scripts/validate_sft_v2.py` is
ground truth after every step. Final rows live in
`data/synthetic/spark_sft_v1/` (see `manifest.json`).

Order: `gen_v2.py` (quotas must exist: `make_sft_v2_quotas.py` + gap/dev100
steps in manifest) -> `rewrite_residuals.py` (Jaccard acceptance) ->
hand-written replacements for stubborn clusters (in data, documented in
manifest) -> `scripts/finalize_sft_v2.py`. Seeds: quotas 20260920, texts
777001, dev100 20260920. Do NOT re-run casually: outputs are committed.
