# STATE — pii-redteam-rl

- **Date:** 2026-10-04
- **Milestone:** M01 done; M02.1–M02.4 done (plan v1.3). M03.2 offline part done;
  M05.2 done. M03.1/M05.1 GPU gate executed on Colab 2026-10-04, re-run 8/8 PASS —
  gate closed. M03.3 detector training next (notebook ready).
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
  Colab/pip fallback exporting the same runtime set). Heavy ML deps (torch CPU,
  transformers, accelerate, peft) live in the locked, opt-in `ml` group:
  `uv sync --group ml` / `uv run --group ml ...` / `make gate-detector`; torch is pinned
  to the PyTorch CPU index via `[tool.uv.sources]`. Never hand-built venvs.
  Convention for LOCAL batch inference: export the checkpoint to ONNX first
  (optimum/onnxruntime) instead of raw PyTorch forward loops — CPU throughput should
  improve several-fold; verify with a measured comparison on the next audit run.
  (Not applied to the running 2026-10-10 audit, which was already in flight.)

- **Option B request bank frozen (M05 groundwork)**: `scripts/build_request_bank.py`
  → `artifacts/request_bank/sampler_seed20260920_n200.jsonl` (200 requests, sha256
  `332a9eb0…dc58c`), drawn from the 16,329 requestable names (74 pseudo dropped),
  re-imported through `load_bank` with pool provenance attached. All 200 entity values
  verified requestable. Same seed + pool version reproduces the bank; regeneration is
  never assumed.

## Interim analysis + D0 notebook prepared (2026-10-04)

- **Name-pool quality census + R07 implemented (owner-approved 2026-10-04)**:
  train_side_pool holds 16,403 raw values; `is_requestable_name` (R07, in
  `src/pii_redteam/requests.py`) drops 74 artifact values (0.45%): 45 digit-containing
  (e.g. `1986rimbiondi`, phone numbers), 12 pipe-joined, 1 email, 8 slash/fillers
  (`-`, `Ajet N/A`), 8 single initials. Hyphenated real names (`Ioan-Ciprian`) stay
  requestable. Enforcement sits in `load_person_name_pool` (drop count recorded in
  pool provenance), so sampler and bank loader reject pseudo values even in
  hand-authored banks. Detector training texts are untouched. The gate notebook's
  sampler call now draws from the requestable subset (7 of the first 8 raw pool
  values are artifacts).
- **`colab/M03_train_d0.ipynb` prepared (not executed)**: trains the D0 candidate on
  train_side rows (positives + ~71% negatives kept), labels via the verified alignment
  seam, per-epoch selection on dev_calibration, best checkpoint by dev span-F1 saved
  to Drive with metrics/curves/errors/manifest. Pilot settings (owner-reviewable):
  seed 20260920, max_length 512, batch 16, 3 epochs (~1 h on T4), lr 5e-5, acceptance
  floor dev span-F1 >= 0.70 + negative-precision guard. Final-eval partition untouched.

## Colab GPU gate executed (2026-10-04, run-20261004-141053, owner-run)

Report from Colab (T4, 15.6 GB VRAM): python 3.13.15, torch 2.11.0+cu130,
transformers 5.18.0, peft 0.21.2, accelerate 1.15.0, repo commit `0ff0568`.

**PASS (7/8):** final-eval lock reproduces on Colab (sha `513335…d39c` identical);
ModernBERT loads for token classification; BIO round-trip 50/50 exact on real rows
(trim fix confirmed on T4); ModernBERT real training step (loss 2.18, grad_sum 1.2e5,
2.28 s, peak 3.11 GB); Qwen3.5-0.8B text-only load via `AutoModelForCausalLM` (bf16);
non-thinking chat template (`enable_thinking=False` accepted); output contract PASS
(JSON `text` field, target name exactly once; generation 7 s, peak 2.82 GB).

**FAIL (1/8):** Qwen LoRA adapter update — `ImportError` from Colab's preinstalled
torchao 0.10.0 (peft/transformers need >0.16.0). Environment issue, not architecture:
the notebook's install cell now upgrades torchao; re-run should confirm.

**Caveats recorded:**
- bf16 was used on a T4 (torch reported bf16 supported); if instability or slow
  generation appears in longer runs, switch to fp16 before debugging anything else.
- The sampler drew pool value `"1963soheila.raimoski"` as a PERSON entity — the
  dataset's GIVENNAME* mask values include username-like digit-prefixed strings.
  Pool-quality policy (filter or keep) is an open owner decision for M05; not
  silently changed here.
- Checkpoints/adapter from the run are on Drive under `run-20261004-141053/`.

## Colab GPU gate re-run (2026-10-04, run-20261004-142042): 8/8 PASS — gate closed

Owner re-ran the notebook after the torchao fix. All checks PASS, including the Qwen
LoRA adapter update (targets auto-discovered: k/o/q/v projections; loss 0.80,
grad_sum 5.9e2). ModernBERT training step 0.33 s (warm cache), peak 4.63 GB. Output
contract again PASS (JSON `text` field, name exactly once). **M03.1/M05.1 complete.**
Report/checkpoints on Drive under `run-20261004-142042/`.

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
- **Permanent ML env + gate script**: `ml` dependency group added (torch>=2.14 CPU,
  transformers>=5.18, accelerate, peft), torch pinned to the CPU index; gate moved to
  `scripts/modernbert_cpu_gate.py` (`make gate-detector`). Reproduced through the
  project env: round-trip 50/50, train step loss 0.99, grad_sum 6.6e4, 2.5 s/step,
  peak RSS 3.3 GB. Report: `artifacts/cpu_gate/modernbert_cpu_report.json`.

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

1. M03.3: run `colab/M03_train_d0.ipynb` on Colab (T4, ~1 h at 3 epochs, Drive cache):
   train_side rows, dev selection, acceptance floors (dev span-F1 >= 0.70 + negative
   precision). Review metrics/errors, then freeze D0 and pin the tested stack.
2. M05: run the frozen sampler bank (Option B) through the generator; owner may also
   author a Spark bank (Option A) for comparison later — same schema, same loader.
3. M07 groundwork: wire the frozen D0 into the reward path once it exists.

## Blockers / open decisions

- D01, D04, D06–D11 open (see `docs/DECISIONS.md`). D02/D03/D05/D12 settled per plan v1.3;
  split/name policies recorded as R05/R06; pseudo-name filter approved and implemented (R07).
  No training, no paid spend, no final-test use.
  M03.1/M05.1 gate closed (8/8 PASS on the 2026-10-04 re-run).
