# STATE — pii-redteam-rl

- **Date:** 2026-09-20
- **Milestone:** M01 done; M02 in progress (plan v1.3 reconciled 2026-09-20).
- **Completed task IDs:** M01.1–M01.3 (see history). M02.1 (PERSON policy in
  `docs/ANNOTATION_POLICY.md`: GIVENNAME*/LASTNAME* → PERSON, TITLE/USERNAME excluded,
  whitespace-adjacent merge). M02.2 (English AI4Privacy files inspected at pinned rev,
  full streaming census, canonical adapter in `src/pii_redteam/data/ai4privacy.py`).
  M05.1-partial (shared request schema §4.1 in `src/pii_redteam/requests.py`: validator,
  prompt renderer, Spark-bank importer with hash+provenance, seeded sampler second path).
- **Completed task IDs:** M01.1 (skeleton: data/detection/generation/rewards/evaluation/accounting
  seams + config loader + fake-model path), M01.2 (uv env with locked pyyaml, seeds,
  resolved-config + manifest per run), M01.3 (schema/unit/integration/smoke tests, resume
  guard, backend contract with reference path + unsupported-config errors). Reuse audit
  written before any code was ported (`docs/REUSE_AUDIT.md`).
  Dependency method: uv only (`uv sync` / `uv run`; `requirements.txt` kept as a
  Colab/pip fallback exporting the same runtime set).

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

- `artifacts/smoke-cpu/{manifest.json,resolved_config.yaml,report.json}` (regenerable, gitignored)
- `uv.lock` (pinned env), `docs/REUSE_AUDIT.md`, `docs/DECISIONS.md`, `README.md`, this file

## Compute usage

- CPU only. No GPU, no downloads, no paid services. Spend: €0.

## Next three actions

1. M02.3: lock split roles — carve the held-out final-evaluation partition (before any
   generator training/prompt authoring from it); record name-overlap policy.
2. M02.4: training-side name pools + small held-out name set from train-file census.
3. M03.1/M05.1: verify Qwen3.5-0.8B text-only loading + adapter-update smoke on Colab
   (free-first gate: feasibility, memory, checkpoint save/resume before any longer run).

## Blockers / open decisions

- D01, D04, D06–D11 open (see `docs/DECISIONS.md`). D02/D03/D05/D12 settled per plan v1.3.
  No training, no paid spend, no final-test use. Uncommitted work in this session:
  docs + `data/` + `requests.py` + fixtures/tests/scripts (commit on owner request).
