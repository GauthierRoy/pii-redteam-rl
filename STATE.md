# STATE — pii-redteam-rl

- **Date:** 2026-10-04
- **Milestone:** M01 done; M02.1–M02.4 done; M03 complete (D0 trained, audited,
  FROZEN 2026-10-10, dev span-F1 0.9633); M05.1/M05.2 done. Next: M05 0.8B pipeline
  proof run, then M06 baseline on Qwen3.5-2B (same model as the RL arm, per D05).
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
  Convention for local inference: prefer running audits on **Colab GPU** (checkpoints
  and dataset cache already live on Drive). ONNX export for local CPU inference is
  deferred — revisit only if Colab access becomes a bottleneck.

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

## M05 0.8B pipeline-proof run executed (2026-10-10, m05-20261010-211905, owner-run)

Run at repo `effe2ea`, T4, torch 2.11.0+cu130, transformers 5.19.0. All locks PASS
(final-eval, request bank, frozen-D0 sha). SFT: 1,728 items (272 dropped >256 tokens),
LoRA 1.08M trainable, epoch losses 1.985 → 1.767 → 1.682, 36 min.

**Results:**

| Metric | Zero-shot | Post-SFT |
|---|---|---|
| valid JSON | 98.5% | **1.5%** |
| name exactly once (word-boundary) | 93.5% | **1.5%** |
| frozen-D0 extra-PERSON flags | 1 / 200 | 2 / 200 |

Diversity (post-SFT, temp 0.8, n=4, 50 reqs): duplicate_rate 0.0, distinct-1 0.765,
distinct-2 0.952, mean 421 chars.

**Interpretation (recorded honestly):**
- **Headline positive:** the BASE 0.8B already follows the output contract (93.5%
  name-exactly-once, 98.5% valid JSON). For pipeline purposes the generator does not
  need SFT to obey the contract.
- **Headline negative — my error, not the model's:** SFT collapsed the contract rate.
  The SFT targets were FULL AI4Privacy records (any single-name-once row), so the
  model learned to emit long bureaucratic PII documents ("Subject: Important Notice:
  Health Insurance…"), which cannot close the JSON inside 150 tokens. This violated
  the plan's own §4.2 ("select SHORT records or extracted snippets") — the fault was
  target selection, not the training.
- **Fix implemented:** `is_short_text` (≤60 words) added to the data adapter
  (+tests); SFT pair selection in the notebook now filters on it. Pool is adequate:
  2,910 train-side rows are single-name-once AND ≤60 words (1,503 at ≤40).
- Frozen-D0 flagging worked as a flag-only signal: outputs contain almost no
  extra person names (1-2 per 200 texts).
- Notebook build consolidated: all fixes (tokenization BatchEncoding, memory
  batch-2/grad-acc-4/256-cap, dirty-GPU guard, allocator conf, no device_map,
  short-text filter) now live in the generator and were verified present in one
  build. Earlier in-place notebook edits had been silently reverted by a
  regeneration — process lesson recorded.

**Token budgets are now measured, not guessed (2026-10-10 follow-up):** on the short
pool (2,910 items, real Qwen tokenizer), JSON-wrapped answers run median 154 / p95 227 /
p99 275 tokens (escaping inflates counts); the old 150 budget would have truncated
54% of the intended targets even after the short-text fix. Full conversations run
median 212 / p99 332. Notebook updated: `GEN_MAX_NEW_TOKENS 150 -> 320` (covers p99),
`SFT_MAX_TOKENS 256 -> 384` (keeps 100% of the pool). Truncation-failure share should
drop to ~1%.

**QLoRA + auto-probed budget (2026-10-10, owner request):** the notebook now loads
Qwen3.5-0.8B as a 4-bit NF4 QLoRA base (double-quantized; ~0.4 GB instead of ~1.6 GB),
trains with `prepare_model_for_kbit_training` (gradient checkpointing on), batch 1 +
gradient accumulation 8, and a runtime **probe** that finds the LARGEST sequence
length surviving a real forward+backward on the actual T4 (candidates 512..8192,
headroom rule: stop at first OOM or >85% peak VRAM; runs after D0 is resident so it
measures the true budget). `SFT_MAX_TOKENS = 0` means "probed, not guessed". For the
current short pool (max 414 tokens) the probed cap is pure headroom - it matters for
the 2B run and later stages. The ≤60-word target filter and the measured 320-token
generation budget stay: they keep outputs inside the short-text scope and the
output contract. Zero-shot/SFT/post-SFT all use the same 4-bit model for
within-run consistency (pipeline proof; cross-quantization comparison is out of
scope for 0.8B).

**Decision needed from owner:** re-run the 0.8B proof with the short-text fix
(another ~1 h Colab), or skip to the 2B M06 baseline (the model the RL arm uses,
per D05) with the fix already in. My recommendation: skip to 2B — the 0.8B run has
already proven the pipeline mechanics (locks, gates, flagging, diversity), and the
SFT-quality question only matters at 2B where the real comparison happens.

## M05 Colab attempt 1: fail-fast triggered, fixed (2026-10-10)

The SFT end-of-turn/prefix gate stopped the run before any training: transformers 5.x
`apply_chat_template(tokenize=True)` returns a `BatchEncoding`, not a list of ids, so
the notebook compared the wrong objects. Fix (verified locally with the real
Qwen3.5-0.8B tokenizer on a real SFT pair): normalize via `out["input_ids"]`; common
prefix is then 100%, label tail ends with `<|im_end|>`. The template's
`return_assistant_tokens_mask` is unsupported (no `{% generation %}` keyword), so
prefix masking remains the method. Notebook cell updated; re-run pending.

## D0 trained and FROZEN (M03.3 done) + negative-FP audit (2026-10-10)

**Run `d0-20261004-142714`** (owner-run on Colab T4, repo commit `da0b7d3`,
torch 2.11.0+cu130, transformers 5.18.0): 28,713 train-side rows (8,191 positive,
negatives kept) / 7,923 dev (2,261 positive); 5,385 steps, 99 min, 7.9 GB peak,
0 truncated spans lost. Curve (dev span P/R/F1): epoch 1 0.9456/0.9672/0.9563,
epoch 2 0.9587/0.9680/**0.9633** (best — saved), epoch 3 0.9591/0.9597/0.9594.

- **M03.3 acceptance vs the 0.70 floor: PASS** (dev span-F1 0.9633; 0.9587 precision
  refutes "labels every capitalized token").
- **Negative-precision floor (2% pilot guess): FAIL** — 129 FPs on 5,662 negative dev
  rows (2.28%) for the frozen epoch-2 checkpoint.
- **Audit (2026-10-10, `scripts/audit_negative_fps.py`, frozen checkpoint, local CPU
  ~40 min):** all 129 FPs classified. 43% are the row's own `privacy_mask` values under
  PERSON-excluded labels (USERNAME 14, SEX 9 — real names the dataset tagged as
  gender, TITLE 8, CITY 6, BOD 6, STATE/STREET/EMAIL/PASS 6); 57% unannotated
  name-like strings (`Yuto Druga`, `Druga`, `Danusa`); only ~6% hard artifacts
  (4 single/empty, 2 separator fragments). Conclusion: negative-row FPs are
  gold/policy noise, not detector hallucination — freeze justified.
- **D0 FROZEN (owner-approved 2026-10-10):** `d0-best` (epoch 2), Drive path
  `pii-redteam-rl/d0-20261004-142714/d0-best`, `model.safetensors` sha256
  `0314ebe36d7f4dc6d784f0e139ff5b9d32633aef1a6f88e764a610a8d8ec43cb`.
  No further detector tuning; D0 is the frozen scorer for M07+.
- Audit execution: future audits run on **Colab GPU** (checkpoint + cache already on
  Drive) — ONNX export deferred, not worth local investment now.

## M05 generator run = 0.8B PIPELINE PROOF (M06 must match the RL arm)

The `colab/M05_generator_sft.ipynb` run is a plumbing/proof run on Qwen3.5-0.8B.
Per D05, the M06 ordinary-generation baseline and the RL arm must use the SAME
model — Qwen3.5-2B. Zero-shot/SFT numbers from the 0.8B run are not comparable
across model sizes and feed no method comparison.

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

1. Run `colab/M05_generator_sft.ipynb` on Colab (T4, ~1 h): bank lock, zero-shot
   contract check, LoRA SFT (full-conversation template, end-of-turn verified,
   ≤2k pairs, 512-token cap), post-SFT re-check, sampled diversity (temp 0.8, n=4),
   and frozen-D0 extra-PERSON flagging (flag only, not reward). This is the 0.8B
   pipeline proof, not a method comparison.
2. M06 baseline: same pipeline on Qwen3.5-2B (same model as the RL arm, per D05) —
   ordinary generation over the frozen bank with the SFT'd 2B generator.
3. M07: wire frozen D0 (sha `0314ebe3…c43cb`) into the reward path; difficulty
   feedback for RL.

## Blockers / open decisions

- D01, D04, D06–D11 open (see `docs/DECISIONS.md`). D02/D03/D05/D12 settled per plan v1.3;
  split/name policies recorded as R05/R06; pseudo-name filter approved and implemented (R07).
  No training, no paid spend, no final-test use.
  M03.1/M05.1 gate closed (8/8 PASS on the 2026-10-04 re-run).
