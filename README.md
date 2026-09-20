# pii-redteam-rl — RL-guided synthetic data for robust PII detection

Planning artifact: `PROJECT_PLAN.md` (v1.2). Agent kickoff: `AGENT_START_HERE.md`.
Reuse audit of the pre-existing `../rl` repo: `docs/REUSE_AUDIT.md`.

## Status

M01 scaffolding on CPU with explicitly mocked models. No training, no model downloads,
no paid compute, no external data calls. All smoke outputs are stamped MOCK and are
excluded from scientific results.

## Quickstart

```bash
uv sync                          # single dependency method; creates .venv + uv.lock
make test                        # 30 unit + integration tests, CPU-only
make smoke                       # fake-model smoke -> artifacts/smoke-cpu/
bash scripts/lint.sh             # ruff + ty via uv
```

Single smoke run:

```bash
python3 scripts/run_smoke.py --config configs/smoke/smoke.yaml --out artifacts/smoke-cpu
```

Re-running into the same `--out` dir fails loudly (resume guard); use a fresh dir or
`resume: true` with the identical model and config.

## Layout

```text
configs/smoke/smoke.yaml      CPU smoke config (fake models, no network)
src/pii_redteam/
  config.py       YAML load / resolve / hash / save (M01.1-M01.2)
  manifest.py     run manifest + checkpoint-resume guard (M01.2-M01.3)
  backends.py     precision/decoder contract, no-speculation reference (M01.3)
  validation.py   span schema + exact-name-once candidate gate
  detector.py     Detector.predict_spans/score_target + FakeDetector
  generator.py    Generator.generate + FakeGenerator
  rewards.py      Reward.score stub: invalid < valid range, components separate
  evaluation.py   exact-span P/R/F1 on fixtures (M04 stub)
  experiments.py  data-to-report smoke workflow
  training/grpo_compat.py  reused TRL-compat shim (see docs/REUSE_AUDIT.md)
scripts/run_smoke.py  smoke CLI
tests/unit/ tests/integration/ tests/fixtures/  stdlib unittest, CPU-only
docs/REUSE_AUDIT.md docs/DECISIONS.md
STATE.md  session continuity log
```

## Hardware / budget

No accelerator needed or used at M01. Heavy training deps
(`torch`, `transformers`, `trl`, `peft`, `vllm`, …) are intentionally not installed
until the M03/M05/M08 pilots. Paid compute: €0 spent, €0 authorized.
