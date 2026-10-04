# STATE — pii-redteam-rl

- **Date:** 2026-10-04
- **Milestone:** M01 done; M02.1–M02.4 done (plan v1.3). M03/M05 next; M05.1-partial done.
- **Completed task IDs:** M01.1–M01.3 (see history). M02.1 (PERSON policy in
  `docs/ANNOTATION_POLICY.md`: GIVENNAME*/LASTNAME* → PERSON, TITLE/USERNAME excluded,
  whitespace-adjacent merge). M02.2 (English AI4Privacy files inspected at pinned rev,
  full streaming census, canonical adapter in `src/pii_redteam/data/ai4privacy.py`).
  M02.3 (split roles locked 2026-10-04: seeded final-eval carve of 1,000 train-file rows,
  `scripts/carve_splits.py` + `src/pii_redteam/data/splits.py`; zero exact/template
  duplicates found; leakage checks asserted). M02.4 (name pools:
  `train_side_pool` 16,403 values + 200 held-out final-eval-unique names,
  `artifacts/splits/name_pools.json`; substring overlap 19/200 recorded).
  M05.1-partial (shared request schema §4.1 in `src/pii_redteam/requests.py`: validator,
  prompt renderer, Spark-bank importer with hash+provenance, seeded sampler second path).
- **Completed task IDs:** M01.1 (skeleton: data/detection/generation/rewards/evaluation/accounting
  seams + config loader + fake-model path), M01.2 (uv env with locked pyyaml, seeds,
  resolved-config + manifest per run), M01.3 (schema/unit/integration/smoke tests, resume
  guard, backend contract with reference path + unsupported-config errors). Reuse audit
  written before any code was ported (`docs/REUSE_AUDIT.md`).
  Dependency method: uv only (`uv sync` / `uv run`; `requirements.txt` kept as a
  Colab/pip fallback exporting the same runtime set).

## Tests actually executed (2026-10-04, M02.3–M02.4 session)

- `PYTHONPATH=src uv run python -m unittest discover -s tests -t .` → 50 tests
  (44 prior + 6 new in `tests/unit/test_splits.py`), OK.
- `bash scripts/lint.sh` → ruff check + format + ty, all pass.
- `uv run python scripts/carve_splits.py` (downloads pinned files to /tmp cache, no
  full-file storage in repo) → train 29,908 rows / 195 quarantined rows / 0 duplicates /
  8,481 person rows; validation 7,946 / 23 / 0 / 2,261. Roles: final_eval 1,000,
  train_side 28,713, dev_calibration 7,923. Held-out names 200/593 candidates.
  Leakage assertions (text disjointness, heldout-pool disjointness) all passed.
  Artifacts: `artifacts/splits/{final_eval.jsonl,final_eval_manifest.json,
  train_side_index.jsonl,validation_index.jsonl,name_pools.json,report.json}`;
  final_eval.jsonl sha256 `513335badf5752a340698d3c7efa7029a8780c51fde2675c80ef4208ce27d39c`.

## Tests actually executed (2026-09-20, plan v1.3 session)

- `make test` → 44 tests (30 M01 + 14 new: ai4privacy adapter, request schema/bank/sampler), OK.
- `bash scripts/lint.sh` → ruff check + format + ty, all pass.
- `uv run python scripts/fetch_ai4privacy_sample.py --max-records 200` → 200 records,
  `artifacts/ai4privacy_sample{,_manifest}.json`.
- `uv run python scripts/build_sft_pairs.py` → rows=200 quarantined=0 multi-or-repeat=174
  pairs=8 → `artifacts/sft_pairs_sample.jsonl`. First pair target verified to contain
  its requested name exactly once.
- Full-file streaming census at pinned rev (not stored): train 29,908 rows / 193,086 masks /
  909 mismatches (0.47%); validation 7,946 rows / 51,184 masks / 113 mismatches (0.22%).
  Mismatch pattern is masking-template leakage in values (e.g. `[GIVENNAME_B(…`), quarantined.

## Artifacts

- `artifacts/splits/` (regenerable from seed 20260920 + pinned revision, gitignored):
  locked final-eval partition + manifests, per-role row indexes, name pools, report
- `artifacts/smoke-cpu/{manifest.json,resolved_config.yaml,report.json}` (regenerable, gitignored)
- `uv.lock` (pinned env), `docs/REUSE_AUDIT.md`, `docs/DECISIONS.md`, `README.md`, this file

## Compute usage

- CPU only. No GPU, no downloads, no paid services. Spend: €0.

## Next three actions

1. M03.1/M03.2: pick ModernBERT token-classification checkpoint; implement tokenizer
   offset alignment, subword labeling, BIO decoding, Unicode tests (train on train_side
   rows only; select on dev_calibration; final_eval untouched).
2. M05.2: wire `train_side_pool` into the sampler/Spark-bank provenance so no generation
   request can carry a held-out or final-eval-only name.
3. M03.1/M05.1: verify Qwen3.5-0.8B text-only loading + adapter-update smoke on Colab
   (free-first gate: feasibility, memory, checkpoint save/resume before any longer run).

## Blockers / open decisions

- D01, D04, D06–D11 open (see `docs/DECISIONS.md`). D02/D03/D05/D12 settled per plan v1.3;
  split/name policies recorded as R05/R06. No training, no paid spend, no final-test use.
  Uncommitted work in this session: `src/pii_redteam/data/splits.py`, `scripts/carve_splits.py`,
  `tests/unit/test_splits.py`, docs + STATE updates (commit on owner request).
