# STATE — pii-redteam-rl

- **Date:** 2026-10-04
- **Milestone:** M01 done; M02.1–M02.4 done (plan v1.3). M03.2 offline part done;
  M05.2 done. GPU-gated work (M03.1/M03.3, Qwen smoke) awaits Colab.
- **Completed task IDs:** M01.1–M01.3 (see history). M02.1 (PERSON policy in
  `docs/ANNOTATION_POLICY.md`: GIVENNAME*/LASTNAME* → PERSON, TITLE/USERNAME excluded,
  whitespace-adjacent merge). M02.2 (English AI4Privacy files inspected at pinned rev,
  full streaming census, canonical adapter in `src/pii_redteam/data/ai4privacy.py`).
  M02.3 (split roles locked 2026-10-04: seeded final-eval carve of 1,000 train-file rows,
  `scripts/carve_splits.py` + `src/pii_redteam/data/splits.py`; zero exact/template
  duplicates found; leakage checks asserted). M02.4 (name pools:
  `train_side_pool` 16,403 values + 200 held-out final-eval-unique names,
  `artifacts/splits/name_pools.json`; substring overlap 19/200 recorded).
  M03.2-partial (tokenizer-agnostic BIO labeling/decoding/truncation in
  `src/pii_redteam/detector.py`, stub-tested offline; real-tokenizer check pending
  Colab). M05.1-partial (shared request schema §4.1 in `src/pii_redteam/requests.py`:
  validator, prompt renderer, Spark-bank importer with hash+provenance, seeded sampler
  second path). M05.2 (train-side pool enforcement in validator/bank/sampler, R06).
- **Completed task IDs:** M01.1 (skeleton: data/detection/generation/rewards/evaluation/accounting
  seams + config loader + fake-model path), M01.2 (uv env with locked pyyaml, seeds,
  resolved-config + manifest per run), M01.3 (schema/unit/integration/smoke tests, resume
  guard, backend contract with reference path + unsupported-config errors). Reuse audit
  written before any code was ported (`docs/REUSE_AUDIT.md`).
  Dependency method: uv only (`uv sync` / `uv run`; `requirements.txt` kept as a
  Colab/pip fallback exporting the same runtime set). Heavy ad-hoc deps (torch etc.)
  go through ephemeral envs: `uv run --no-project --with <pkgs> ...` — never hand-built
  venvs, never the locked project env.

## Tests actually executed (2026-10-04, ModernBERT CPU gate session)

- **M03.1 partially verified on this laptop (CPU, not Colab)** — ephemeral uv env
  (repo lock untouched): `uv run --no-project --with transformers --with torch
  --default-index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
  python /tmp/mb_cpu_gate.py` → torch 2.14.1+cpu, transformers 5.18.0.
  Script: throwaway copy of the notebook's ModernBERT cell over 50 real train-side rows
  (final-eval ids excluded). Report: `artifacts/cpu_gate/modernbert_cpu_report.json`.
  - Load: `answerdotai/ModernBERT-base` loads as `ModernBertForTokenClassification`
    (fresh head: classifier weight/bias newly initialized, as expected).
  - Round-trip: initially 29/50 off-by-one — root cause: ModernBERT's BPE tokenizer
    prefixes a word's first token with the preceding space. Fix:
    `trim_span_whitespace` in `src/pii_redteam/detector.py` (+2 tests) applied to
    recovered spans before comparison. After the fix: **50/50 exact**.
  - Training step (batch 4 × 128 tokens): loss finite (1.55, random head), grad_sum
    8.6e4 > 0, ~2.2 s/step, peak RSS ~3.3 GB on 16-core CPU. Full D0 training stays
    planned for Colab; per-step cost makes laptop-only training slow but possible.
- `PYTHONPATH=src uv run python -m unittest discover -s tests -t .` → 66 tests
  (64 + 2 trim tests), OK. `bash scripts/lint.sh` → all pass.

## Tests actually executed (2026-10-04, M03.2/M05.2 session)

- `PYTHONPATH=src uv run python -m unittest discover -s tests -t .` → 64 tests
  (50 prior + 10 alignment + 4 pool-enforcement), OK.
- `bash scripts/lint.sh` → ruff check + format + ty, all pass.
- M03.2 (offline part): `src/pii_redteam/detector.py` gained tokenizer-agnostic
  `bio_labels_from_spans` / `spans_from_bio_labels` / `truncate_labels` operating on
  char-offset sequences (the HF fast-tokenizer `return_offsets_mapping` interface).
  Round-trip, Unicode, subword, special-token, orphan-I, mid-boundary, and truncation
  behavior all covered by `tests/unit/test_alignment.py` with stub offsets. The real
  ModernBERT tokenizer still needs verification on Colab (M03.1) — mBERT arrays remain
  unused. D0 training (M03.3) not started.
- M05.2: `src/pii_redteam/requests.py` enforces R06 — `load_person_name_pool`,
  `validate_request(..., person_name_pool=...)` (rejects held-out/final-eval names),
  `load_bank(..., person_name_pool=...)` (records pool sha256/size in the manifest),
  `seeded_sampler(..., allowed_names=...)` (raises on out-of-pool names).

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

1. Run `colab/M03_M05_gpu_gate.ipynb` on Colab (free T4, `Runtime -> Run all`): it
   mounts Drive, clones this repo, rebuilds + asserts the locks, then executes the
   M03.1 ModernBERT gate and the M05.1 Qwen3.5-0.8B smoke. Copy the printed summary
   and the `gpu_gate_report.json` Drive path back here as evidence.
2. M03.3: train D0 on train_side rows (tokenizer-aware labels via the verified
   alignment seam), select on dev_calibration, freeze before any RL.
3. M05.2-done follow-up: author the first Spark request bank from `train_side_pool`
   (or run the seeded sampler) and import it through the pool-enforcing loader.

## Blockers / open decisions

- D01, D04, D06–D11 open (see `docs/DECISIONS.md`). D02/D03/D05/D12 settled per plan v1.3;
  split/name policies recorded as R05/R06. No training, no paid spend, no final-test use.
  `colab/M03_M05_gpu_gate.ipynb` prepared but NOT yet executed — no M03.1/M05.1 evidence
  exists until the owner runs it on Colab; model IDs verified to exist on the Hub
  (`answerdotai/ModernBERT-base`, `Qwen/Qwen3.5-0.8B`, hybrid `Qwen3_5ForConditionalGeneration`).
  Uncommitted work in this session: the notebook + this STATE update.
