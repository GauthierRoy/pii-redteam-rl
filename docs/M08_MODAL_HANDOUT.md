# M08.3 Modal runs: handout (2026-11-13 / 2026-11-14)

Branch `feat/m08-rl-smoke`. All changes below are uncommitted.

## 1. Bottom line

- The RL pipeline works end to end on Modal: TRL GRPO + LoRA on Qwen3.5-2B against the frozen D1 reward, colocated vLLM generation, checkpoint resume, 840 steps in 138 minutes on one L4.
- The 840-step policy is **not usable as adversarial data**. It collapsed onto a single letter template (`Dear <name>, … Sincerely, <signer>`) for every requested context type. D1 does miss names in that template, but the data would cover one weakness out of ten context types.
- Cause: the D18 reward (`r_prob = 1 − mean gold-label probability`, invalid = −1) has no term for context adherence or diversity, and validity v2 does not check either. The reward saturated near 0.9, so later steps gave little signal.
- Decision needed from the owner: whether to amend D18 (see section 7). I made no reward change.

## 2. Setup

| Item | Value |
|---|---|
| Launcher | `scripts/modal_rl.py` (entrypoints `bench`, `resume_test`, `long`) |
| Modal profile / volume | `gogo979r` / `pii-redteam-rl` |
| GPU | L4, about $0.80/h; container with 4 CPU + 16 GiB about $1.12/h |
| Model | `Qwen/Qwen3.5-2B`, bf16 LoRA (r 16, alpha 32, all-linear), no 4-bit |
| Reward | D18 unchanged; sampling temperature 1.0, top_p 1.0 |
| Detector | frozen D1 `d1-frozen-d1v2-20261110-075731`, sha256 verified on the volume |
| Configs | `configs/m08_rl_bench.yaml`, `configs/m08_rl_long.yaml` (new) |

Runs must use `modal run --detach` (under `caffeinate -i`). Without `--detach`, the first benchmark died on a local client disconnect and cost about $0.115 for nothing.

## 3. Timing benchmark (10 GRPO steps, L4, bf16)

| Arm | Step time | Generation | Peak GPU | Note |
|---|---|---|---|---|
| `hf` (HF generate) | 12.9 s | 8.7 s | 5.9 GB | baseline |
| `fla` (flash-linear-attention kernels) | 13.0 s | n/a | n/a | no gain, 83 s first-step compile |
| `vllm_names` (vLLM + weight-name rename) | **7.06 s** | 2.5 s | 11.1 GB | adopted |
| `vllm_lm` (vLLM `language_model_only`) | 7.36 s | n/a | n/a | no benefit |

- HF generation was about two thirds of a step. The remaining ~4.6 s is training, reward and weight sync.
- The fast-path kernel warning ("hybrid-attention kernels fall back to slow PyTorch code") matters less than expected: installing fla changed nothing. vLLM's native Gated DeltaNet Triton kernels are what gave the speedup.

## 4. The vLLM fix for Qwen3.5

- Qwen3.5-2B is natively multimodal (`Qwen3_5ForConditionalGeneration`, has `vision_config`). `AutoModelForCausalLM` loads it text-only, with parameter names `model.*`.
- vLLM 0.33.0 loads the multimodal class and expects `model.language_model.*` (vLLM issue #36275). TRL's weight sync then fails with `no module or parameter named 'model' in Qwen3_5ForConditionalGeneration`.
- Fix in `src/pii_redteam/training/rl_smoke.py`: `to_multimodal_qwen_name` renames `model.X` to `model.language_model.X` (leaving `model.language_model.*`, `model.visual.*` and `lm_head.weight` alone). `align_vllm_names_with_multimodal_checkpoint` patches TRL's `VLLMGeneration._fix_param_name_to_vllm` to apply it. Enabled by the config key `vllm_multimodal_names: true`.
- Tried and dropped: vLLM `language_model_only` (does not fix the names), vLLM Transformers backend (cannot run linear-attention layers).
- TRL officially supports vLLM 0.22–0.32. 0.33 gave warnings only. Pin or re-check on upgrade.

## 5. Other code changes

- `TimedGRPOTrainer` records per-update generation seconds; `timing_summary` adds `step` and `generation` medians to the report. TRL already logs `step_time`, so no separate timer.
- `tests/integration/test_grpo_update_path.py`: tests for the timing, the language-model-only wrapper and the name renaming (229 tests pass; ruff and ty clean at the last check).
- `colab/M08_rl_smoke.ipynb`, `src/pii_redteam/cli.py`, `configs/m08_rl_smoke.yaml`, `EXPERIMENTS.yaml`, `STATE.md` also carry earlier uncommitted edits from the smoke run work.
- Not updated yet: `STATE.md` and `EXPERIMENTS.yaml` do not contain the Modal results.

## 6. Checkpoint resume

`resume_a` (20 steps) followed by `resume_b` (30 steps, resumed from the latest checkpoint) worked: training continued from the saved step.

## 7. Long run (`long-20261114-004948`)

Config: `configs/m08_rl_long.yaml`; lr 2e-5, beta 0.02, 8 rollouts x 4 prompts per step, `scale_rewards: false`, vLLM colocated at 0.3 memory, checkpoint every 50 steps.

Result: stopped by the 135-minute cap at step 840 (planned 1000). Median step 9.05 s (about 9.5 s all-in), median generation 3.76 s, peak GPU 11.1 of 23.7 GB, wall 8281 s, about $2.58. All final checks pass except "training reached the step cap", which fails by design here.

Training curves: reward rose from −0.19 to about 0.8–0.9; D1 miss rate rose to 0.78–1.0; KL rose from 0.22 to 0.39 on average (0.71 at step 713); entropy about 1.0–1.2; grad norm shrank to 0.07–0.16.

### Samples: 256 per side, 32 held-out requests

| | pre | post |
|---|---|---|
| valid | 216/256 | 254/256 |
| mean reward | −0.06 | 0.86 |
| D1 detects the name (p_gold > 0.5) | 203/216 | 5/254 |
| text starts with "Dear" | 33/216 | 253/254 |
| mean words | 20.5 | 43.8 |
| unique texts | 216 | 254 |

- Legitimate: fluent English, the requested name always present, no gibberish, no repeated text.
- Degenerate: every request becomes a formal letter, including `form_record_line`, `list_or_table_row`, `chat_message` and `third_person_report`. 244/254 contain a closing such as "Sincerely". The `context_type` is ignored.
- Before RL, 15% of samples already opened with "Dear"; RL amplified an existing mode.
- The 5 texts D1 still detects are mixed (two appointment reminders, one "Hello <name>" greeting, two other "Dear" letters), so five samples are too few to say more than that D1 misses names in this letter format.
- Validity v2 does not reject extra person names, and nothing checks context adherence.

## 8. Spend

| Item | Amount |
|---|---|
| Posted for 11-13 | $0.59 |
| Long run (launcher estimate) | about $2.58 |
| Estimated total | about $3.66 |
| Ledger stop / budget | $4.70 / $5.00 |

Billing lags. Re-check with `LC_ALL=C modal billing report --start 2026-11-13` (the French locale breaks `awk` sums). No apps are running. The night-watch cron was deleted.

## 9. Open decisions and next steps

1. **Reward and validity (owner decision, D18 amendment):** add a context-adherence check or penalty and a diversity term; consider the bounded log-odds reward. Stop earlier (about 100–200 steps) or raise beta, since reward saturated.
2. **Sampling:** top_p 1.0 versus 0.95, unchanged.
3. **Before M09 labelling:** filter texts with extra person names; the `Surname_A` placeholder gap (R07) and M06 prerequisites remain open.
4. **Remaining Modal budget** is about $1.3, enough for a short test of a modified reward, not a full rerun.
5. **Docs:** record sections 3–8 in `STATE.md` and `EXPERIMENTS.yaml` if wanted.
6. **Security:** the Modal API token was pasted into the chat. Rotate it (Settings > API Tokens) and re-run `modal token set`.

## 10. Artifacts

On the Modal volume `pii-redteam-rl`: `/runs/long-20261114-004948/` holds `config.yaml`, `run.log`, `summary.json` and the child run `m08-long-20261114-005016/` (`rl_smoke_report.json` about 1.3 MB, `samples_pre.jsonl`, `samples_post.jsonl`, `requests_signal.jsonl`, `checkpoints/`, `adapter/`). Benchmark and resume runs sit alongside it under `/runs/`.
