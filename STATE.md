# STATE — pii-redteam-rl

- **Date:** 2026-09-20
- **Milestone:** M01 done (scaffolding + CPU smoke); M00 decisions drafted, all D01–D11 open.
- **Completed task IDs:** M01.1 (skeleton: data/detection/generation/rewards/evaluation/accounting
  seams + config loader + fake-model path), M01.2 (uv env with locked pyyaml, seeds,
  resolved-config + manifest per run), M01.3 (schema/unit/integration/smoke tests, resume
  guard, backend contract with reference path + unsupported-config errors). Reuse audit
  written before any code was ported (`docs/REUSE_AUDIT.md`).
  Dependency method: uv only (`uv sync` / `uv run`; `requirements.txt` kept as a
  Colab/pip fallback exporting the same runtime set).

## Tests actually executed

- `make test` (→ `PYTHONPATH=src uv run python -m unittest discover -s tests -t .`) → 30 tests, OK.
- `make smoke` → 8 candidates (8 valid), fake-detector P=0.000 R=0.000 on 3 fixtures. MOCK.
- `bash scripts/lint.sh` (ruff check + format --check + ty via uv) → all pass.

## Artifacts

- `artifacts/smoke-cpu/{manifest.json,resolved_config.yaml,report.json}` (regenerable, gitignored)
- `uv.lock` (pinned env), `docs/REUSE_AUDIT.md`, `docs/DECISIONS.md`, `README.md`, this file

## Compute usage

- CPU only. No GPU, no downloads, no paid services. Spend: €0.

## Next three actions

1. Owner review: confirm D02 (language/domain), D03 (data policy), D04 (budget gate process).
2. M02 start: annotation policy + split/lock design on synthetic fixtures (no real data yet).
3. S00 start: verify DFlash sources/licenses + GPU FP8/speculative feature matrix on paper.

## Blockers / open decisions

- All D01–D11 open (see `docs/DECISIONS.md`). No training or data ingestion until D02/D03 land.
