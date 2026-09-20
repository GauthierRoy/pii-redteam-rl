# Reuse audit: `../rl` (rl-mini / TinyR1-Zero) → `pii-redteam-rl`

- **Date:** 2026-09-20
- **Source inspected:** `/home/gauthier/code/rl` (README: "TinyR1-Zero — GRPO Reasoning on a 1.5B Model")
- **Source revision:** `d56910d` (`git log` head; history reviewed, working tree not touched)
- **Method:** read-only inspection. No file in `../rl` was modified. No code was copied into this
  repository before this audit was written.
- **Target milestone:** M01 (minimal reproducible repository + CPU smoke tests). No training,
  no model downloads, no paid compute.

## 1. Existing components relevant to this project

| Source path | What it is | Relevance to PII project |
|---|---|---|
| `src/grpo_compat.py` (18 lines) | Version-tolerant `GRPOConfig` constructor: passes kwargs, drops unknown keys with a warning | **Directly relevant (M08).** TRL renames/drops args over time; this shim is the cheapest insurance. Only working infra component with no task-specific logic. |
| `src/train_countdown.py` (120 lines) | Config-driven GRPO entry: YAML → tokenizer → synthetic dataset → LoRA → `GRPOTrainer` → `train(resume…)` | **Pattern relevant.** Config-driven launch, `resume` flag wired to `resume_from_checkpoint`, Qwen3 `enable_thinking` template patch, `mask_truncated_completions`, `use_liger_loss` passthrough, fail-fast on missing local model dir. Task body (countdown) is not relevant. |
| `src/train_grpo.py` (103 lines) | Same pattern for GSM8K (reward-weight wiring, LoRA, `use_vllm` flag) | **Pattern relevant, superseded** by `train_countdown.py` (newer: resume, thinking patch, liger). GSM8K loader is not relevant. |
| `src/sft_format.py` (100 lines) | Programmatic SFT cold-start: solver-built traces → `SFTTrainer` → `merge_and_unload` → merged fp model | **Pattern relevant (M05).** Solver-built cold-start data + `SFTConfig` compat shim + merge step. Countdown solver itself is not relevant. |
| `src/rewards_countdown.py` (91 lines) | TRL-compatible reward funcs, free-text candidate scan, tiered shaping (2.0/0.3/0.1), DAPO-style overlong band, format tiers | **Convention relevant (M07).** Signature shape (`completions, <columns>, **kwargs → list[float]`) and the overlong-band idea transfer. All countdown semantics stay behind. |
| `src/envs/countdown.py` (185 lines) | Reverse-generation synthetic task generator + exact-once checker + echo-proof prompt | **Lesson relevant, code not.** Reverse-gen (solution→problem guarantees solvability) and echo-proof worked examples are the transferable ideas (M05/M09). Nothing PII-specific exists here. |
| `MEMORY.md` run log | Failure analysis: cold-start format failure, Qwen3 pre-closed-`<think>` ceiling, echo hack, truncation masking, monitor gates | **Relevant as operational knowledge** (SFT-before-RL, correctness-only early, `frac_reward_zero_std`/entropy/KL gates). Not code. |
| `src/rewards.py` | GSM8K/MATH `math-verify` correctness + `<think>/<answer>` format + word-count length penalty | Not relevant. Math verifier and think/answer tags have no PII counterpart. |
| `src/eval.py` | vLLM `pass@1`/`maj@8`/avg-length on GSM8K/MATH-500/SVAMP | Not relevant as code. Batch-generate-then-aggregate shape is a vague future model for M04/M10 only. |
| `configs/*.yaml` (11 files) | Countdown/GSM8K hyperparams; several hardcode `/content/drive/MyDrive/rl-mini/…` outputs | Not relevant. No config is portable (absolute Drive paths, countdown hyperparams). |
| `notebooks/countdown_colab.ipynb` | Colab runbook for the countdown runs | Not relevant. Task-specific; provider-launcher pattern to redo later per plan §9. |
| `outputs/smoke/` | Scratch checkpoints/completions from smoke runs | Not relevant. Ephemeral by the source `.gitignore`. |
| `scripts/lint.sh`, `.pre-commit-config.yaml`, `pyproject.toml` (ruff/ty), `Makefile`, `.gitignore`, `requirements*.txt` | Tooling/env scaffolding | **Relevant as scaffolding.** Lint/type pipeline, ignore rules, Colab install notes (vLLM CUDA-12 pin, torchao conflict). |

## 2. What to reuse, adapt, or leave behind

### Reuse (near-verbatim; no redesign)

- `src/grpo_compat.py` → `src/pii_redteam/training/grpo_compat.py`, with **one functional change**:
  lazy `trl` import inside the builder so the module *imports* on CPU-only M01 environments
  (the original top-level `from trl import GRPOConfig` breaks import without the training stack).
  Construction still requires `trl` at call time. Logic otherwise verbatim.
- Scaffolding: `scripts/lint.sh` shape, `.pre-commit-config.yaml`, ruff/ty sections of
  `pyproject.toml`, `.gitignore` training-output rules, `Makefile` target shape
  (`install`, `test`, `smoke`, `lint`), requirements pinning *approach* (incl. the Colab
  `vllm<0.20` CUDA-12 lesson recorded for S00, not installed at M01).

### Adapt (take the pattern, write new task code; do not port bodies)

- Config-driven YAML launch + LoRA block + `resume` flag + `mask_truncated_completions`-style
  trainer knobs (`train_countdown.py:35-114` pattern) → new PII trainer entry in M08 only.
  At M01 this yields just the config loader + run-manifest/resume-guard scaffolding.
- Qwen3 `enable_thinking=False` template patch (`train_countdown.py:48-60`) → recorded as a
  known generator-tokenizer hazard for M05; re-implemented only if the chosen generator needs it.
- Reward-function *signature convention* and overlong-band *shape* (`rewards_countdown.py`)
  → new `Reward.score` interface returning separately-logged components
  (validity / difficulty / repetition / KL), per plan M07.3. No countdown tiers ported.
- SFT cold-start → merge flow (`sft_format.py:43-94`) → M05 design note only.
- Reverse-generation + echo-proof-example lessons (`envs/countdown.py`, `MEMORY.md`) → M05/M09
  design notes. The countdown generator/checker stay behind.

### Leave behind (explicitly not copied)

- `src/rewards.py`, `src/rewards_countdown.py`, `src/envs/countdown.py`, `src/eval.py`,
  `src/train_grpo.py` / `src/train_countdown.py` bodies, `src/sft_format.py` body.
- All 11 `configs/*.yaml` (Drive-absolute paths, countdown/GSM8K values).
- `notebooks/`, `outputs/`, W&B/Gradio/vLLM runtime wiring, `uv.lock`.

## 3. Tests supporting reuse vs missing validation

- **Existing tests in `../rl`: none.** There is no `tests/` directory, no unit test, no
  integration test. Stated verification is manual: run logs summarized in `MEMORY.md`
  (format-reward 0/40 cold start, 0.5B 300-step failure, Qwen3.5 partial run), notebook gate
  cells, and `outputs/smoke/` checkpoints. None of this is machine-checkable from a fresh clone.
- **What that means for each reuse item:**
  - `grpo_compat.py` — untested against multiple TRL versions (its entire purpose). It is 18
    lines of introspection; risk is low but real. Needs a new unit test with a fake config class
    (added at M01: `tests/unit/test_grpo_compat.py`, stdlib-only, no `trl` import).
  - Trainer/SFT patterns — no update-path test exists (policy/reference log-probs, masks, KL,
    frozen-detector params, adapter saves). M08 must add one; nothing to inherit.
  - Reward patterns — no fixture-based reward tests exist. M07 requires deterministic reward
    tests + failure fixtures from scratch; countdown expectations do not transfer.
  - Data/split/leakage, span alignment/decoding, metric fixtures, checkpoint-resume,
    backend-error paths — no coverage at all. All are M01–M04 new work (resume-guard and
    backend-error tests are part of the M01 gate below).
- **M01 acceptance mapping:** the smoke suite added here covers deterministic fixtures,
  manifest/resume guard, backend unsupported-config errors, and isolated output dirs —
  exactly the gaps the source repo never closed at scaffolding level.

## 4. Assumptions incompatible with the new project

1. **Verifiable math reward ≠ detector-feedback reward.** Countdown/GSM8K rewards check exact
   equations/answers with `math-verify`/AST eval. The PII reward needs a *frozen* detector,
   subword-to-character offset alignment, truncation-window checks, and exact-name-once
   validity (plan M07.1–M07.2). No component for any of this exists in `../rl`.
2. **Single decoder task ≠ detector+generator loop.** Source trains one Qwen policy on one
   synthetic task. The project needs a ModernBERT(-compatible) token-classifier detector
   (M03), an SFT generator (M05), RL against the frozen detector (M08), matched augmentation
   arms incl. generate-and-filter + continued-training control (M06/M09/M10), and disparity
   analysis (M11). No detector, annotation, split, or retraining code exists to reuse.
3. **Serving setup ≠ RL rollout setup.** Source runs `use_vllm: true/false` as a simple flag.
   The plan requires FP8/AWQ/speculative/DFlash to stay *out* of on-policy rollouts until
   behavior-policy probabilities and trainer corrections are validated (M08.2, §6A guardrails).
   The flag pattern transfers; the assumption that serving == rollout does not.
4. **No data governance.** Source loads GSM8K or generates from thin air; there are no split
   locks, leakage checks, name pools, provenance manifests, or final-test isolation (M02/M04.4).
   Nothing here may touch final evaluation data, by design absent from the source.
5. **Absolute Drive paths and Colab-session outputs** (`/content/drive/MyDrive/rl-mini/…`,
   `outputs/…`) contradict the reproducibility contract (resolved configs + env metadata +
   isolated run dirs, M01.2–M01.3). New configs use relative `artifacts/` outputs.
6. **Budget model.** Source assumes free/`<$10` single GRPO runs. The project needs a ledger
   separating SFT, rollouts (incl. rejected), filtering, detector retraining, and DFlash
   trace/training costs under a ~€50 exploratory envelope (plan §9). No accounting code exists.
7. **Qwen-specific template patch** (`enable_thinking=False`, pre-closed `<think>`) must not
   be applied blindly to the future generator/detector tokenizers; it is a documented hazard,
   not a default.
8. **No license file** was found in `../rl`; checkpoint/model licenses (Qwen, GSM8K, MATH,
   SVAMP) are not recorded there. The new project records provenance per source from M02
   (`docs/DATA_CARD.md`); nothing is inherited implicitly.

## 5. Minimal migration proposal

1. Copy `src/grpo_compat.py` → `src/pii_redteam/training/grpo_compat.py` (lazy-`trl` adaptation
   noted above) + `tests/unit/test_grpo_compat.py` (fake-config unit test, no heavy deps).
2. Copy scaffolding only: `.gitignore` rules, `scripts/lint.sh`, `.pre-commit-config.yaml`,
   ruff/ty `pyproject.toml` sections, `Makefile` target shape.
3. Write new M01 modules from scratch against the plan §7 interfaces
   (`Detector.predict_spans/score_target`, `Generator.generate`, `Validator.validate`,
   `Reward.score`, `Evaluator.evaluate`) with an explicit fake-model CPU path; add the
   generation-backend contract (precision/decoder options, no-speculation reference,
   `UnsupportedBackendError`) and the manifest/resume guard.
4. Add CPU smoke (`configs/smoke/smoke.yaml` + `scripts/run_smoke.py`) and stdlib-`unittest`
   suites (unit + integration/resume/isolation); keep heavy deps
   (`torch`, `transformers`, `trl`, `peft`, `vllm`, …) out of the M01 install.
5. Do not copy anything listed under "Leave behind". Do not port hyperparams, prompts,
   rewards, datasets, notebooks, or outputs. Reference `../rl/MEMORY.md` lessons in commit
   messages/design notes instead of duplicating code.
6. Verify: `python3 -m unittest discover -s tests -v` green on CPU; smoke writes an isolated
   `artifacts/` run with manifest + report; re-running with a different `model_id` into the
   same dir fails loudly (resume guard); `git status` in `../rl` clean (untouched).
