# STATE — pii-redteam-rl

- **Date:** 2026-11-15
- **Now (M08.3):** three Modal RL runs done, none usable as adversarial data yet; reward amended twice
  (D20 opening-share guard, D21 person judge). Next: one 400-step pilot with both
  (`configs/m08_rl_person.yaml`). Details in the first section below. Model names changed on
  2026-11-14 (D19, `docs/GLOSSARY.md`).
- **Milestone:** M07b done: target detector (`d1v2-20261110-075731`, epoch 2) FROZEN 2026-11-10 and the
  M07a probe go/no-go passes against it at temperature 1.0 (M08 gate formally open).
  Base detector frozen 2026-10-10 (dev span-F1 0.9633). M08.1 done as a recommendation (TRL
  `GRPOTrainer`, CPU-smoke-tested, owner accepted; verl reviewed from docs only, kept as a
  later option). M08.2 done on a tiny model (update path, masks, groups, zero-variance,
  controlled learning, target detector reward parity); D18 records the owner's trainer/`scale_rewards`/
  dependency decisions and the RL dataset adapter exists. Next: M08.3 smoke-run preparation
  (RL config, RL-scale request bank, cost estimate/cap/checkpoint/shutdown plan); M06
  baseline and the base-2B diversity check still to close.
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

## M08.3 Modal runs: collapse, then a non-person exploit (2026-11-14/15)

Three runs on one Modal L4, same seed, bank, sampling and optimizer; only the reward differs.
Full numbers in `EXPERIMENTS.yaml`; decisions D20 and D21.

| Run | Reward | Steps | Target detector finds the name (held-out, post) | Name is a person (judge) | Verdict |
|---|---|---|---|---|---|
| `long-20261114-004948` | D18 | 840 | 5/254 | 254/254 | One real weakness, over-exploited: every text is a "Dear <name>" letter |
| `pilot-opening-20261114-103332` | D18 + hard opening guard (0 above 0.20) | 200 | 229/250 | 238/250 | No collapse, nothing learned |
| `pilot-opening-20261114-124322` | D18 + soft opening guard (0.25 -> 0.50) | 400 | 7/240 | 4/240 | Invalid: name used as a place or brand |

Pre-RL reference on the same 32 held-out requests: the target detector finds 203/216 names;
the name is a person in 200/216.

- **Long run.** Reward saturated by step 200 (0.79; detector miss rate 0.85). The letters are
  genuine person mentions that the target detector misses, so the weakness is real, but it is
  one weakness out of ten context types.
- **Context adherence is not checked (decided against, D20).** Three API judges recovered the
  labelled context type of `short_eval_v2` rows only 25-37% of the time (top-3: 67-77%). The
  labelled texts themselves only loosely follow their type ("support ticket" rows are prose
  about a ticket).
- **Opening-share guard (D20).** For each valid text: share of the last 256 valid texts written
  for other requests that start with the same word. On the long run's samples: at most 0.15
  pre-RL, 0.99 post-RL. The hard cut at 0.20 removed the reward of 14-29% of valid texts in
  training (the base share swings around 0.15) and nothing was learned in 200 steps. The soft
  version keeps the full reward up to 0.25 and scales it to zero at 0.50.
- **Non-person exploit.** With the soft guard, learning was slow for 250 steps, then reward
  jumped to 0.77 by step 400: "a great taco spot near the Olimpico", "the nearest Mulaton
  store", "the El Bez clinic". The validity rules never checked that the name is a person.
- **Person judge (D21).** `zai-glm-5-3-flash` on the Mistral API, asked once per valid text
  inside the reward; a non-person use is invalid. It kept 248/250 persons and rejected 234/236
  non-persons on labelled texts; about 2.5-5 s per 32 calls when measured from a laptop.
- **Spend.** Long run about $2.58, pilots about $0.61 and $1.13 (launcher estimates; Modal
  billing lags). Monthly Modal budget raised to $30 by the owner on 2026-11-14.
- **Not done.** No diversity measure beyond the opening word. The extra-person-name gap
  (signers such as "Dr. Sarah") is still open. Whether the "Dear" weakness is the greeting,
  the length or the sign-off has not been isolated.

## M08.2 update-path validation done on a tiny model (2026-11-13; trainer = TRL `GRPOTrainer`)

**Code added (uncommitted):** `src/pii_redteam/training/rl_reward.py` (`TargetDetectorReward`, a TRL
reward callable: strict-JSON contract + one word-boundary name mention via the probe's
`classify_sample`, reward `r_prob = 1 - mean detector probability over the name tokens`,
invalid/truncated completions = -1.0, below the valid range [0, 1]; logs `valid_rate`,
`truncated_rate`, `true_miss_rate` and per-completion `status` / `p_target_mean`) and
`src/pii_redteam/training/detector_scorer.py` (`DetectorScorer`: freezes the detector, calls the probe's
unchanged `_score_with_detector`). Only `r_prob` is offered: the probe's `r_logit` is unbounded
(+-10), so the -1.0 invalid reward could outrank valid completions.

**Tests actually run (all passing):**
- `PYTHONPATH=src uv run python -m unittest tests.unit.test_rl_reward` -> 12 unit tests
  (fake scorer, no torch): statuses, reward ordering, truncation via `completion_ids`,
  conversational completions, all-invalid group, metrics denominators, logging, group order.
- `tests/integration/test_grpo_update_path.py` -> 9 tests, ~12 s on CPU, real
  `GRPOTrainer` + LoRA on `hf-internal-testing/tiny-random-LlamaForCausalLM` (opt-in; skipped
  without trl; run with `uv run --no-project --with torch --with transformers --with trl
  --with peft --with accelerate --with datasets --with pyyaml ...`). Verified:
  prompts are repeated G times contiguously and per-group advantages sum to 0; completion
  mask is 1 up to and including the first EOS and 0 / pad after it; only LoRA tensors move
  (base weights bit-identical); a constant-reward group gives advantage 0, grad_norm 0 and no
  parameter change; an all-invalid group through `TargetDetectorReward` gives reward -1, zero std and no
  update; with `beta > 0` KL is ~0 at step 1 and positive later (reference = initial attacker,
  adapter disabled); with `beta = 0` no reference log-probs are computed; a frozen detector
  is untouched by scoring during training; controlled reward learning (reward = -length)
  raises the mean reward by > 2 tokens in 30 steps.
- `PII_BASE_DETECTOR_PATH=<d0-best> ... python -m unittest tests.integration.test_target_detector_reward_parity` ->
  2 tests: for the 5 recovered real base-2B pilot generations `TargetDetectorReward` + `DetectorScorer` with
  frozen base detector (sha-checked) give exactly `1 - recorded p_target_mean` (tolerance 6e-4), and a
  truncated completion of a scorable text is invalid. Skips without the checkpoint.
- `make test`: 198+ tests OK (new integration tests skip in CI); `bash scripts/lint.sh` passes.

**Test-design lessons recorded:** the tiny random Llama never emits EOS (all completions
equal length, zero reward variance), so its output layer gets an EOS bias; it also has
near-uniform logits, so the learning test scales the output layer 20x (an unscaled model
showed essentially no learning at any learning rate).

**Open risk found for M08.3 (decided in D18, to be verified in the smoke run):** TRL computes advantages as
`(r - mean) / (std + 1e-4)`. With the continuous `r_prob` reward and a near-saturated target detector
(p ~ 0.99+), group reward differences can be ~1e-3, which TRL's default `scale_rewards="group"`
inflates to O(1) advantages, i.e. noise amplification, while `frac_reward_zero_std` stays 0
because std is not exactly zero. Owner decision (D18): `scale_rewards=False` for the smoke
run, `reward_std` watched, group scaling kept as a one-flag comparison on real target detector rewards.

**M08.3 preparation done 2026-11-13:** `requires-python` is now `>=3.11` and `trl>=1.17` is in
the locked `ml` group (the 9 GRPO tests pass in the locked env). `src/pii_redteam/training/
rl_dataset.py` turns validated v2 request records into `{prompt, name, request_id}` rows with
the P1 prompt (7 unit tests). The only committed v2 bank has 20 requests, far too few for RL:
an RL-scale bank (names from the train-side pool only, disjoint from comparison/dev requests)
is still to be built.

**NOT validated yet:** the same properties on Qwen3.5-2B with the real tokenizer/chat
template; log-prob agreement between sampling and training under 4-bit QLoRA; T4 memory and
speed; target detector rewards on real generations at scale (needs the base-2B P1 samples, which live on
Drive); end-to-end TRL run with the real reward on a GPU.

## M08.1 trainer evaluation: TRL `GRPOTrainer` smoke-tested on CPU (2026-11-12, owner accepted TRL for the pilot)

**Recommendation:** use TRL's `GRPOTrainer` (critic-free GRPO) with a LoRA adapter on
Qwen3.5-2B, with a custom reward function that calls frozen target detector. It is the simplest option
that already has the reward-function interface, group sampling, masks, KL and the metrics
M08.3 needs; writing a trainer from scratch is not justified by anything seen so far.

**What was actually run (laptop CPU, scratch script, not committed; deleted after):**
real `Qwen/Qwen3.5-0.8B` + `peft` LoRA (r=8, q/k/v/o), `GRPOTrainer` through the repo's
`build_grpo_config`, 4 prompts x 4 generations, 24 new tokens, temperature 1.0,
`enable_thinking=False`. Resolved versions: torch 2.15.0, transformers 5.22.2, trl 1.17.1
(the repo lock has transformers 5.18.0 and no trl; this was an ephemeral env, the lock is
untouched).
- **Mechanics work:** generation, reward call, advantage, loss, optimizer step, logging.
  Logged `reward`, `reward_std`, `frac_reward_zero_std`, `completions/clipped_ratio`,
  `entropy`, `clip_ratio/*`, step time.
- **Zero-variance case behaves as expected:** with a constant reward (name-once, 4/4
  correct) `frac_reward_zero_std` = 1, loss 0, grad_norm 0, LoRA parameters unchanged
  (abs change 0.0). This is exactly the regime ~89% of groups would be in at target detector's ~1.4%
  miss rate (see the target detector section), so group size/reward design is the real M08 risk.
- **Real update confirmed with a varying reward** (toy: shorter is better): grad_norm 1.40,
  `frac_reward_zero_std` 0, 48 trainable LoRA tensors changed (abs change 29.5).

**Findings that matter for the build:**
- Pass `processing_class=AutoTokenizer.from_pretrained(...)` explicitly: for Qwen3.5 TRL
  otherwise builds a multimodal `AutoProcessor`, which needs torchvision + pillow and is
  pointless for a text-only task.
- TRL defaults differ from the plan: `beta=0.0` (no KL, no reference model), `loss_type=dapo`,
  `scale_rewards=group`. The plan requires tracking KL, so set `beta > 0` explicitly
  (with a peft adapter the reference is the adapter-disabled model, no second copy).
- `chat_template_kwargs={"enable_thinking": False}` is accepted by the config.
- CPU timing is meaningless for planning (~130 s/step, reference-kernel fallback because
  `flash-linear-attention` and `causal_conv1d` are not installed); GPU speed with those
  kernels is untested.

**verl considered (docs/PyPI review only, NOT run, 2026-11-12):** verl 0.9.1 is built for
multi-GPU clusters with vLLM/SGLang rollouts and supports LoRA, GRPO, custom reward
functions and KL loss. Against this project's stack it has concrete frictions: it requires
Python < 3.13 (Colab here runs 3.13), pins transformers `>=5.5.3,<5.11` (the Colab runs
used 5.18–5.22) and torch 2.11, needs Ray plus vLLM 0.24 or SGLang, its install/Docker path
targets CUDA 13 on modern GPUs (flash-attn wheels; T4 is Turing, no bf16, untested), the
`hf` rollout option is listed but its status is unclear, and its Qwen3.5 examples are 35B
(no dense 0.8B/2B example seen). TRL also offers vLLM rollouts, so the speed gap is smaller
than it looks. Re-evaluate verl only if the TRL pilot shows a real learning signal and
M10-scale runs on Modal/Vast need more throughput; that would be a separate decision.

**NOT tested / unverified:** Qwen3.5-2B; 4-bit QLoRA under RL (Unsloth docs warn that
QLoRA on Qwen3.5 has larger-than-usual quantization differences, which matters when
behavior and reference log-probs must agree; prefer fp16/bf16 LoRA if the 2B fits on the T4
and decide with a measured memory probe); T4 memory and throughput; HF `generate` speed for
G=8–16 rollouts; completion-mask and prompt/group-id correctness (M08.2); target detector as the reward
inside the trainer; `trl` is not yet in the `ml` dependency group or `uv.lock`.

## Target detector FROZEN (M07b.3/M07b.4 done, owner-approved 2026-11-12; freeze run d1v2-freeze-20261110-081446)

Source: freeze manifest `freeze_manifest.json` from the Drive archive
`d1-frozen-d1v2-20261110-075731`, read 2026-11-12. Repo commit `676ce54` (on
`feat/m07b-freeze-d1`; contains `target_detector_aug_v1` and `short_eval_neg_v1`). Training run
`d1v2-20261110-075731` on Colab, ModernBERT token classifier (O / B-PERSON / I-PERSON),
Target detector = base detector + `target_detector_aug_v1` augmentation. Archive integrity verified locally: model,
config and both tokenizer files match the sha256 values in the manifest.

- **Frozen checkpoint:** `d1-epoch2`, `model.safetensors` sha256
  `ebfe4c693958c84cdb093d1a55b535e51f1ba0563a702e69e3c97bf5a36a4143`.
  Epoch 3 (sha `51237386…1f21a4`) was the alternative.
- **Selection rule (pre-registered, dev only):** `fewest_dev_neg_fp_rows` among epochs
  within 0.01 dev F1 of the base detector (0.9633). Epoch 2: dev F1 0.9605, 67 name-free dev rows with
  a prediction. Epoch 3: dev F1 0.9636, 100 rows. Chosen epoch 2; recomputed rule matches
  the training run's own choice.
- **Locks:** final-eval partition lock reproduces
  (`513335badf5752a340698d3c7efa7029a8780c51fde2675c80ef4208ce27d39c`); frozen base detector
  `0314ebe3…8ec43cb`; M07a probe config unchanged (`261ec868…`).
- **All required manifest checks pass.** Gates:

| Gate | Base detector | Target detector | Threshold | Result |
|---|---|---|---|---|
| Final-eval span-F1 | 0.9539 | 0.9530 | drop ≤ 0.01 | pass (drop 0.0009) |
| Final-eval precision / recall drop (positive = worse) | – | – | ≤ 0.01 each | pass (precision −0.0026, recall 0.0042) |
| Name-free final-eval rows with a false positive (710 rows) | 9 | 9 | relative increase ≤ 30% | pass (1.27%, allowed 1.65%) |
| Short name-free messages with a false positive (95 gated rows) | 0 | 3 | ≤ 5% (PROPOSED) | pass (3.16%) |
| M07a probe, temperature 1.0 | – | miss rate 1.38% | band [0.01, 0.30] | go |
| M07a probe, temperature 0.7 | – | miss rate 1.34% | band [0.01, 0.30] | no-go (`duplicates`) |


**Caveats recorded, not hidden:**
- **Sparse reward risk.** target detector misses only ~1.4% of base-2B generations, just above the
  band floor. With group-relative advantages, a group of 8 contains at least one miss
  only ~10.5% of the time (16: ~20%), so ~89% of groups would have zero variance.
- **New short-text false positives (read 2026-11-12).** base detector had 0/95 on short name-free
  messages, target detector has 3/95 (reproduced locally with the frozen weights and the freeze's own
  `predict_spans`; counts match the manifest: place 1, org_product 2). All three are
  name-like proper nouns, not ordinary words: `Sintra` (Portuguese town, id
  `short-eval-neg-v1-0085`), `Clearline Freight` (company, `-0116`), and `bridge`, a
  sub-word fragment of the company `Ironbridge Manufacturing` (`-0051`). No
  homograph, calendar or role-only row fires. The 5% threshold is still labelled
  PROPOSED, not pre-registered.
- **short_eval, base detector vs the target detector (local CPU rescoring 2026-11-12; both checkpoints sha-verified,
  freeze's own `predict_spans` at `676ce54`, data shas match the locks).** The M07b.3 gain
  gate (`short_eval_recall_gain_min` 0.10 on `short_eval_v2`) **passes: +0.5733**.

| Set | Model | Precision | Recall | F1 | exact / partial / miss |
|---|---|---|---|---|---|
| `short_eval_v2` (150) | Base detector | 0.7590 | 0.4200 | 0.5408 | 63 / 18 / 69 |
| `short_eval_v2` (150) | Target detector | 0.9868 | 0.9933 | 0.9900 | 149 / 0 / 1 |
| `short_eval_deepseek_v1` (50, same writer as `target_detector_aug_v1`, D17) | Base detector | 0.6875 | 0.4400 | 0.5366 | 22 / 9 / 19 |
| `short_eval_deepseek_v1` (50) | Target detector | 1.0000 | 1.0000 | 1.0000 | 50 / 0 / 0 |

  Target detector fixes 68 of the base detector's 69 `short_eval_v2` misses and all 19 on the deepseek slice; no row
  Base detector hit is lost by the target detector (zero regressions). The only target detector miss is `short-eval-v2-0131`.
  Short name-free rows: base detector 0/95 flagged, target detector 3/95 (ids above). The target detector is nearly saturated on
  short messages, consistent with the ~1.4% probe miss rate. The two tokenizers differ
  only in `tokenizer_config.json` default keys (`tokenizer.json` is byte-identical).
- **Temperature.** Only temperature 1.0 passes the probe (0.7 fails on duplicates);
  M08 generation stays at 1.0 unless re-justified.
- **EXPERIMENTS.yaml:** freeze run `d1v2-freeze-20261110-081446` registered from the
  manifest (2026-11-12), with the local base detector vs the target detector rescoring attached. The training run
  `d1v2-20261110-075731` is not registered yet; it needs its own report JSON from Drive.

## short_eval_neg_v1: name-free short messages for the target detector freeze (2026-11-09)

Target detector run `d1-20261108-145253` was only scored on short messages that each contain one
name, so its false-positive rate on name-free short text was unmeasured; on AI4Privacy
dev its name-free false positives rose 129 → 162. New eval set
`data/synthetic/short_eval_neg_v1.jsonl` (111 rows, sha256 `a0ffa9ec…9069f`), written
by GLM-5.3 via Mistral Vibe, every row with no PERSON span and a `distractor` group:
place 20, org_product 23, calendar 23, role_only 29 (95 gated rows) and homograph 16
(report-only; owner decision 2026-11-09: words that are also names, used in their
ordinary sense, never gate the freeze). 150 raw rows → 111 after a deterministic
reuse cap on invented names copied from the prompt examples (owner: "drop some of the
duplicates"), with the two capitalized subject-line homographs (idx 128, 133) kept by
owner decision, and 1 validator drop. All 150 raw rows read by hand: no person names.
Validator: `sft_synth_v2.validate_negative_row`. Rebuild with
`PYTHONPATH=src python3 scripts/finalize_short_eval_neg_v1.py`. The freeze config on
`feat/m07b-freeze-d1` must pin this sha.

## Interim analysis + base detector notebook prepared (2026-10-04)

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
- **`colab/M03_train_base_detector.ipynb` prepared (not executed)**: trains the base detector candidate on
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

## M05b 2B run executed overnight (2026-10-10/11, m05-20261010-222751, owner-run)

Run at repo `a2b0d0a`, T4, transformers 5.19.0, QLoRA 4-bit. All locks PASS
(GPU fresh 15.5 GB, final-eval, bank, frozen base detector). Probe: max training tokens **512**
(next candidate exceeded the 85% headroom rule); 1,987/2,000 pairs trained
(13 answers > 288-token answer cap; 0 conversation skips). SFT 2h06m, losses
1.867 → 1.667 → 1.593. Per-epoch saves worked as designed.

| Metric | 2B zero-shot | 2B post-SFT |
|---|---|---|
| valid JSON | **100%** | 86.5% |
| name exactly once | **97.5%** | 75.5% |
| truncated | 0 | few |
| extra names (base detector flags) | 2/200 | 4/200 |
| diversity (post-SFT) | not measured | dup 0.0, distinct-1 0.679, mean 670 chars |

**Failure anatomy (20 recorded post-SFT failures):** 8 finished-but-invalid JSON
(model ends turn before closing, or malformed content), 9 valid JSON but the
requested name is missing (occ 0), 3 truncated. NOT a budget problem this time —
genuine behavior drift: the SFT'd model writes AI4Privacy form-letters
("Subject: Important Update on Your Health Insurance Plan…", bracketed
placeholders like "Dear [Lucas],") and loses the exact-name contract in a quarter
of cases.

**Decisive negative result:** dataset-derived SFT degrades contract compliance at
BOTH sizes tested (0.8B: 93.5% → 1.5% with long targets; 2B: 97.5% → 75.5% with
budget-aligned targets). The target style (synthetic form-letter records) is the
cause, not budgets or tokenization. Meanwhile the BASE 2B already exceeds the
contract zero-shot (100% JSON / 97.5% once).

**Recommendation (owner decision):** adopt the BASE Qwen3.5-2B as the attacker —
M06 ordinary-generation baseline AND the RL arm's starting state — and drop the SFT
stage, recording this evidence as the justification (the plan accepts valid negative
results). Caveat to close first: only the SFT'd model's diversity was measured;
run the base-2B diversity check (sampled, temp 0.8) before freezing, since generic
zero-shot outputs could be less varied. If diversity is acceptable, M07 proceeds
directly: frozen base detector as reward scorer, RL from base 2B.

## D13 settled: two-stage reward-model structure (owner decision, 2026-10-11)

M07a verdict accepted (REAL SHIFT: probe == eval pipeline everywhere; base detector misses
~53% of short-conversational names vs 2.4% on dev). Recorded as D13 in
docs/DECISIONS.md and a new **M07b** milestone in PROJECT_PLAN.md — no silent
threshold edits:

- **Stage 1 (M07b, no RL):** `short_eval` (held-out names + style-independent
  slice of ~50 human-written/other-LLM messages); target detector = base detector + plain short-message
  augmentation (Spark SFT data and/or base-2B generations); report base detector vs the target detector on
  short_eval AND final-eval (no-regression check); re-run the M07a probe against
  Target detector (same go/no-go table); freeze the target detector.
- **Stage 2 (M08–M10):** frozen target detector is the RL reward model; the four arms
  (plain generation / generate-and-filter / Spark data / RL) train the retrained detector.

Next actions below updated to this structure.

## M07a scorer sanity verdict recorded (2026-10-11, PR #5)

Verdict **REAL SHIFT, not a scorer bug** (docs/PROBE_SCORER_SANITY.md): probe scorer
== base-detector-training eval pipeline on 124/124 dev gold spans and 32/32 pilot texts;
controlled dumps show base detector itself assigns p(B-PERSON) 0.02–0.08 to short-message
names; specials/id2label/alignment/single-token checks all clean. Regression guard:
`tests/integration/test_probe_scorer_regression.py` (+ fixture; 5 fixed dev rows
must exact-hit; the "Hey Lucas…" tripwire is pinned to the frozen base detector's sha). The 1–30%
miss-rate band is KEPT: base detector's pilot miss rate (~53%) fails it on the high side —
The base detector is too weak on short messages to be the RL target, so RL moves to the frozen target detector
per the D13 two-stage structure above.

## Owner decision (2026-10-10): skip 0.8B re-proof; go straight to 2B, overnight-capable

The notebook (`colab/M05_attacker_sft.ipynb`, now "M05b") is switched to
**Qwen/Qwen3.5-2B** — the model the M06 baseline and the RL arm must share (D05).
Hardened for multi-hour/overnight runs: the report and the LoRA adapter are flushed
to Drive after every stage AND after every SFT epoch, so a Colab disconnect costs at
most one epoch. Same QLoRA + auto-probed token budget + budget-aligned target filter
+ truncation-aware reporting as the 0.8B build. The 0.8B run remains the pipeline
proof; its numbers are not comparable to 2B.

## M05 0.8B pipeline-proof run executed (2026-10-10, m05-20261010-211905, owner-run)

Run at repo `effe2ea`, T4, torch 2.11.0+cu130, transformers 5.19.0. All locks PASS
(final-eval, request bank, frozen base detector sha). SFT: 1,728 items (272 dropped >256 tokens),
LoRA 1.08M trainable, epoch losses 1.985 → 1.767 → 1.682, 36 min.

**Results:**

| Metric | Zero-shot | Post-SFT |
|---|---|---|
| valid JSON | 98.5% | **1.5%** |
| name exactly once (word-boundary) | 93.5% | **1.5%** |
| frozen base detector extra-PERSON flags | 1 / 200 | 2 / 200 |

Diversity (post-SFT, temp 0.8, n=4, 50 reqs): duplicate_rate 0.0, distinct-1 0.765,
distinct-2 0.952, mean 421 chars.

**Interpretation (recorded honestly):**
- **Headline positive:** the BASE 0.8B already follows the output contract (93.5%
  name-exactly-once, 98.5% valid JSON). For pipeline purposes the attacker does not
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
- Frozen base detector flagging worked as a flag-only signal: outputs contain almost no
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
headroom rule: stop at first OOM or >85% peak VRAM; runs after the base detector is resident so it
measures the true budget). `SFT_MAX_TOKENS = 0` means "probed, not guessed". For the
current short pool (max 414 tokens) the probed cap is pure headroom - it matters for
the 2B run and later stages. The ≤60-word target filter and the measured 320-token
generation budget stay: they keep outputs inside the short-text scope and the
output contract. Zero-shot/SFT/post-SFT all use the same 4-bit model for
within-run consistency (pipeline proof; cross-quantization comparison is out of
scope for 0.8B).

**Target selection is now budget-aligned token-space (2026-10-10, owner request):**
the ≤60-word proxy was replaced by a direct criterion: a pair trains only if its
JSON-wrapped ANSWER tokenizes to ≤ `GEN_MAX_NEW_TOKENS - 32` (288 tokens at the
current 320 budget) and its conversation fits the probed training budget. Measured
on the full pool (3,806 single-name-once rows, no word filter): answers median 151 /
p99 273 / max 357; the 288-token cap keeps 99% (3,782). Every trained pattern is
therefore reproducible within the generation budget. `is_short_text` remains a repo
utility but is no longer the notebook's criterion.

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

## Base detector trained and FROZEN (M03.3 done) + negative-FP audit (2026-10-10)

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
- **Base detector FROZEN (owner-approved 2026-10-10):** `d0-best` (epoch 2), Drive path
  `pii-redteam-rl/d0-20261004-142714/d0-best`, `model.safetensors` sha256
  `0314ebe36d7f4dc6d784f0e139ff5b9d32633aef1a6f88e764a610a8d8ec43cb`.
  No further detector tuning; base detector is the frozen scorer for M07+.
- Audit execution: future audits run on **Colab GPU** (checkpoint + cache already on
  Drive) — ONNX export deferred, not worth local investment now.

## M05 attacker run = 0.8B PIPELINE PROOF (M06 must match the RL arm)

The `colab/M05_attacker_sft.ipynb` run is a plumbing/proof run on Qwen3.5-0.8B.
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
    8.6e4 > 0, ~2.2 s/step, peak RSS ~3.3 GB on 16-core CPU. Full base detector training stays
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
  unused. Base detector training (M03.3) not started.
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

## spark-sft-v1 dataset done (2026-10-15, feat/spark-sft-data)

- **Synthetic SFT v2 complete**: `data/synthetic/spark_sft_v1/` — train 1195
  (idx 0–1149 + 45 gap-fill over 15 tail cells), dev 100 (balanced 100-subset
  of 150 quota rows), short_eval 300 (held-out names only). Manifest with
  sha256 per file. Registered in `EXPERIMENTS.yaml` as `spark-sft-v1`
  (`kind: dataset-synthetic`, milestone M05).
- **Validation**: `scripts/validate_sft_v2.py` over all authored batches —
  0 failures on all 7 checks, 0 exact dupes, 0 near-dupes (>0.7);
  top-10 opening share 0.0752, name-at-start 0.2508.
- **Method note**: seeded quotas (20260920, regenerated bit-identically after
  a /tmp wipe destroyed all staging) + committed template-bank generator
  (`scripts/sft_v2_gen/`, texts seed 777001) with synonym paraphrase and
  Jaccard acceptance rewrite. Base detector extra-name flags **pending** (frozen
  checkpoint unavailable at finalize time; flag script ready).
- **Tests**: 94 unit tests OK. `ruff check` + `ruff format` pass; `ty` clean
  in new files (6 remaining warnings pre-exist in untouched main files).
- No scorer/prompt-code changes. No training.

## spark-sft-v1 relabeled template_paraphrase; short_eval deprecated (2026-10-15, fix/spark-sft-v1-relabel)

- **Audit verdict (post-merge of PR #7)**: the texts are NOT individually
  LLM-written — they are outputs of the seeded template-bank + synonym-paraphrase
  generator (`scripts/sft_v2_gen/`, texts seed 777001). Known artifacts: filler
  padding to hit word counts ("Filed promptly. Agreed unanimously. Confirmed
  Friday."-style fragments in ~71% of train rows), near-identical rows (same
  template, only name/number varied), synonym artifacts ("speeds yet collapse").
- **Relabel (D16)**: row `provenance` changed `spark_authored` ->
  `template_paraphrase` across train/dev (+ deprecated short_eval); validator,
  finalizer, generator, tests updated to enforce the new label. New sha256s in
  `data/synthetic/spark_sft_v1/manifest.json` (train `f62df178…`, dev
  `cb33d950…`). The `EXPERIMENTS.yaml` `spark-sft-v1` entry is left untouched
  (append-only registry records the as-executed 2026-10-15 state; its sha256s
  refer to the pre-relabel files).
- **Usability**: spark-sft-v1 may be used ONLY as the M10 **template baseline**
  arm — a baseline to measure the gap between template-paraphrase data and
  LLM-authored/generated data. It is not Spark/LLM-authored-quality data.
- **short_eval_v1 DEPRECATED** -> `data/deprecated/spark_sft_v1_short_eval_v1.jsonl`
  (+ `data/deprecated/README.md`): contaminated — 298/300 rows share a
  name/number-normalized 8-word shingle with train rows (same template bank,
  e.g. `short_eval-0009` vs `train-0004/0007/0010`). Must never be used for
  evaluation. The replacement (M07b.1) must use an independent method per D13.
- PR #7 is NOT reverted; the merged history and original artifacts are preserved.
- **Validator re-run with the new content checks** (`feat/sft-v2-content-checks`,
  2026-10-15, all 1595 spark-sft-v1 rows incl. the deprecated short_eval):
  - Original 7 checks: 1595/1595 pass (schema, exact_once, spans, refusal_meta,
    pii_containment, length, position). 0 exact dupes, 0 near-dupes (>0.7).
  - New **filler check** (repeated sentence / >= 3 sentences under 4 words /
    repeated 3-gram): **803/1595 rows FAIL** (train 607/1195, dev 42/100,
    short_eval 154/300).
  - **Template reuse** (>= 8 consecutive shared words): train 1179/1195,
    dev 98/100, short_eval 298/300 rows share a run with another row.
  - **Cross-split overlap with eval rows** (hard-fail rule): 4698 train/dev x
    short_eval row pairs share an 8-word run — must be ZERO; short_eval_v1 is
    therefore unusable, consistent with its deprecation.
  - **spaCy extra-person pass** (`en_core_web_sm` 3.8.0, CPU, ~1 min): extra
    PERSON entities flagged in train 78/1195, dev 3/100, short_eval 17/300
    (flag-only; LLM-judge extra-person + person-sense pass NOT run — needs
    Qwen3.5-0.8B weights; script ready at `scripts/person_check_v2.py`).

## short_eval_v2 authored (2026-10-15, feat/short-eval-v2, STEP 2)

- **150 rows, all texts written directly by the LLM, one by one** — no template
  bank, no paraphrase/synonym tools, no filler (provenance
  `llm_authored_direct:glm-5.3`). Authored drafts preserved in
  `data/synthetic/short_eval_v2_batches/` (5 files x 30); finalized by
  `scripts/finalize_short_eval_v2.py` with code-computed spans.
- **Names**: 150 distinct held-out names, one row each (never training-side,
  never final-eval content).
- **Coverage**: 15 rows per context type (all 10); positions start/middle/end
  40 each + signature 30; length buckets 50 each (20-30 / 31-45 / 46-60);
  name_form first 68 / surname 68 / full 14.
- **Validation** (hard asserts in the finalizer, all passed):
  `validate_row` 150/150 (schema, exact-once, spans, position, length, hygiene
  incl. filler checks); 0 exact dupes; 0 near-dupes (>0.7); **0 within-set
  template-reuse pairs (>= 8-word run)**; **0 8-word overlap with spark_sft_v1
  train/dev and the deprecated short_eval_v1**. top-10 opening share 0.0667;
  name-at-start 0.2667.
- **Independent extra-person check**: spaCy `en_core_web_sm` pass (CPU,
  ephemeral env) flagged 9/150, all false positives ("Cheers" sign-off cues,
  "Mill Lane", "Keys", "Volunteer", "metadata") — no real extra person names.
  LLM-judge extra-person + person-sense pass NOT run (needs Qwen3.5-0.8B
  weights; `scripts/person_check_v2.py` ready).
- **Artifacts**: `data/synthetic/short_eval_v2.jsonl` (sha256
  `0bee4dc3431f558fb5f076d9ac0acc68d8cddf5313646935100ce3b23e68749a`; row 0016 relabeled 2026-11-08, see below) +
  `data/synthetic/short_eval_v2_manifest.json`. Human-written slice slot
  reserved: `data/synthetic/short_eval_deepseek_v1.jsonl` (renamed from
  `short_eval_human_v1`, see D17; 50 rows, DeepSeek, PR #11; provenance
  `llm_authored_direct:deepseek-v1-flash`, same schema/checks).
- **Name overlap with short_eval_deepseek_v1 (PR #11)**: after writing
  independently (as required), a names-only comparison found 38 shared names.
  25 single-name rows were revised to unused held-out names (reviewed
  substitutions only; texts otherwise untouched). Remaining overlap: **13
  multi-token names — forced**: the pool holds only 14 usable multi-token
  held-out names and both eval files use them, so zero overlap is impossible
  without changing name_form coverage.
- **Row 0016 relabeled (2026-11-08)**: "explained the clash to Taekwondo, both
  clubs agreed" read as a club but was labeled PERSON. The name is now the
  unused held-out `Petrea`; the sentence is otherwise unchanged. Previous sha256
  `bd890d64ede13d76cb19a8850992f174564da31cd80e3c1402fa13d1f70968de`. No detector
  has been scored on this set yet.
- **Provenance label note**: plan text says `llm_authored_direct:glm-5.2`;
  this session's active model is **GLM-5.3** (config `active_model =
  glm-5.3`), so the accurate label `llm_authored_direct:glm-5.3` is used.
- **Validator quirk found (not changed)**: `_SIGNOFF_CUE`'s second
  alternative omits `signed`, so a "Signed,\nName" close is not detected as a
  signature; rows were authored with the accepted cue set (Regards/Best/
  Thanks/Cheers/--). Proposed fix recorded for a future decision; no
  pre-committed threshold changed.

## target_detector_aug_v1 authored (2026-10-15/16, data/d1-aug-deepseek-v1, M07b.2)

- **2000 rows, all texts written directly by the LLM, one by one** (provenance
  `llm_authored_direct:deepseek-v1-flash`) — stage-1 augmentation training data
  for the target detector, split `sft_train`; no templates, paraphrase/synonym tools, or filler.
  Authored drafts preserved in `data/synthetic/target_detector_aug_v1_batches/` (40 files x
  50); finalized by `scripts/finalize_target_detector_aug_v1.py` with code-computed spans.
- **Authoring plan**: batch assignments (context/position/length bucket/name)
  pre-generated deterministically (seed 20261015); each text authored around
  the assigned entity, then validated per batch before acceptance.
- **Names**: training-side pool only (`artifacts/splits/name_pools.json`
  `train_side_pool`); 1512 distinct names across 2000 rows; **zero overlap with
  held-out (final-eval-only) names**. Not evaluation data.
- **Coverage**: 200 rows per context type (all 10); positions start/middle/end/
  signature 500 each; length buckets 680 (20-30) / 680 (31-45) / 640 (46-60);
  name_form first 480 / surname 560 / full 960.
- **Validation** (hard asserts in the finalizer, all passed):
  `validate_row` 2000/2000 (schema, exact-once, spans, position, length, hygiene
  incl. filler checks); 0 exact dupes; 0 near-dupes (>0.7); **0 within-set
  template-reuse pairs (>= 8-word run)**; **0 8-word overlap with
  `short_eval_v2`, `short_eval_deepseek_v1` (PR #11 draft), the deprecated
  short_eval_v1, and `spark_sft_v1` train/dev**. top-10 opening share 0.0280;
  name-at-start 0.2500.
- **Artifacts**: `data/synthetic/target_detector_aug_v1.jsonl` (sha256
  `0d46c41f5d9257abc214ddfe190772ccee8a9069c6baff0511538aaea1c4cdf7`; was
  `d1_aug_v1.jsonl`, sha256 `c16e037dd978c5305e510474653ac573a6dde9e560a945da6428c8f31609eb3d`,
  before the 2026-11-14 rename changed the row-id prefix) +
  `data/synthetic/target_detector_aug_v1_manifest.json`. PR on `data/d1-aug-deepseek-v1`,
  **stacked on `feat/short-eval-v2`** (the `llm_authored_direct:<writer>`
  provenance validator support lands there).
  PR #13 therefore merged into `feat/short-eval-v2`, not main (its base was never
  retargeted after #10 merged); re-landed on main by `fix/land-d1-aug-v1`
  (2026-11-08), finalizer re-run with an identical sha256.

## Eval-set person check (2026-11-08, `results/short_eval_person_check/`)

- Person-sense judge (Qwen3.5-0.8B, local CPU) is NOT a usable gate: on hand-written
  controls it said "person" for 10/10 person uses and 8/10 clear non-person uses.
  Its 0 rejections on the 200 eval rows are not evidence the labels are right.
- Manual read of all 200 target names: one likely mislabel, `short-eval-v2-0016`
  ("Taekwondo" reads as a club). PR #18 relabels it to `Petrea`.

## Artifacts

- **`EXPERIMENTS.yaml` (repo root)** — append-only registry of every executed run:
  run_id, milestone, where, repo commit, environment, artifacts location, pass/fail
  checks, key metrics, outcome (negative results kept as first-class entries).
  Validated by `tests/unit/test_experiments_registry.py`. Update from each run's own
  report JSON, never from memory.
- `artifacts/splits/` (regenerable from seed 20260920 + pinned revision, gitignored):
  locked final-eval partition + manifests, per-role row indexes, name pools, report
- `artifacts/smoke-cpu/{manifest.json,resolved_config.yaml,report.json}` (regenerable, gitignored)
- `uv.lock` (pinned env), `docs/REUSE_AUDIT.md`, `docs/DECISIONS.md`, `README.md`, this file

## Compute usage

- CPU only. No GPU, no downloads, no paid services. Spend: €0.

## Next three actions (post target detector freeze)

Previous list (M07b.1, M07b.2, M07b.4) is done: `short_eval_v2`, `target_detector_aug_v1`, target detector freeze
and re-probe.

1. **M07b paperwork (done 2026-11-12 except one item):** false-positive rows read, freeze
   run registered, base detector vs the target detector `short_eval` gate computed locally (+0.5733, pass). Only the
   training run's `EXPERIMENTS.yaml` entry remains (needs its report JSON from Drive);
   it does not block M08.
2. **M08.3 preparation:** done: `scale_rewards` decision (D18), request-bank -> TRL dataset
   adapter, `trl` in the locked `ml` group. Still to do: RL config, RL-scale request bank,
   then a capped Colab smoke run with the cost estimate, hard cap, checkpoint path and
   shutdown plan the handoff file requires. The smoke run must report, per group, how many
   rollouts target detector misses (sparse-reward check) before the options below are chosen.
3. **Close the M06 prerequisites:** base-2B diversity check (sampled, temperature 0.8)
   and the ordinary-generation baseline on the frozen request bank, so M08.4 has
   its comparison arms.

## M08.3 smoke run EXECUTED on Colab T4 (2026-11-13, run m08-smoke-20261113-170803)

Registered in `EXPERIMENTS.yaml`. 30 GRPO steps, 688 s (22.9 s/step), peak memory 3.8 of 15.6 GB.
Fixed on the way (each found by running on the GPU, not locally): summary read a field the
sampled rows lack; TRL casts QLoRA adapters to bf16 which the fp16 grad scaler cannot unscale
(trainable params now cast to fp32 on fp16 runs); Colab hides subprocess output (now streamed).

Findings (read-only analysis of the saved samples, target detector rescoring matched stored p within 3e-6):
- Valid rate 0.64 -> 0.90 (json 49 -> 2, truncated 5 -> 0, too_short 12 -> 3, name_missing 23 -> 12,
  name_dup 1 -> 7). About 93% of the reward gain is validity; detector misses 2 -> 1 of 256.
- The 0.64 baseline is far below M07b (0.92): pre-RL had fences (19), think-tag leaks (6), `{}` (5),
  truncations (5) that M07b almost never saw. Cause not isolated: top_p 1.0 vs 0.95, 4-bit fp16 vs
  fp32 CPU, different request set.
- Style drift toward "Dear Name," openers (19% -> 35% of valid texts), where target detector is weakest
  (mean p 0.83 vs 0.97 for other openings). Persistent target detector weak requests: Jabar, Swetha, Cirilla.
- Extra person names (sign-offs, "Dr. X") in 5.5% -> 8.7% of valid texts (not significant); target detector sees
  only about a third of them. Validity v2 does not reject them: they must be filtered before
  any generated text is labelled (the extra name would be labelled O) in M09.
- grad_norm NaN on steps 1-4 and 15 with finite loss, KL ~0 until step 6: consistent with fp16
  grad-scaler skipped updates (about 25 real updates). Not verified. bf16 GPUs avoid it.
- Minor: RL bank contains placeholder name `Surname_A` (R07 filter gap); report `rule` provenance
  label is the held-out rule text (label only, the code loads `train_side_pool`).
- Not shown: any detector-evasion learning, durable effect of the style drift (one seed, one stage),
  `resume_from_checkpoint`, scale_rewards comparison.

## M08.3 smoke run prepared (2026-11-13, before the run)

- **Reward validity fixed before any GPU run:** `TargetDetectorReward` first used the v1 classifier, under
  which a 2-word text such as "Hi Lucas" is valid and could earn reward ~1 if the target detector missed it
  (reward hacking toward degenerate text). It now uses `validity_v2.classify_sample_v2`, the
  rules P1 passed under in M07b (D15). The parity test was updated: v2-valid pilot rows still
  equal `1 - recorded p` (real base detector, passes), the 2 of 5 pilot rows that are too short now get -1.
- **Code:** `training/rl_bank.py` (seeded RL request bank: train-side names only, disjoint from
  the listed comparison banks, balanced context types; 7 unit tests), `training/rl_smoke.py`
  (pre-RL signal sampling, capped GRPO with wall-clock stop and checkpoints, post-RL sampling,
  health checks, every raw sample saved), CLI `rl-smoke`, `configs/m08_rl_smoke.yaml`,
  `colab/M08_rl_smoke.ipynb` (thin launcher; last cell releases the runtime).
- **Executed locally (CPU):** the whole runner end to end on SmolLM2-135M-Instruct with the base detector as a
  stand-in detector, a fake name pool and 2 steps, plus the wall-clock stop (1 of 3 steps).
  The 135M model never produced valid JSON, so reward was constant -1: this proves wiring and
  that the health checks flag a dead run (they FAILED as they should), not learning.
- **NOT executed:** Qwen3.5-2B, 4-bit QLoRA under TRL, T4 memory and speed, `resume_from_checkpoint`,
  the real name pool (`artifacts/` is not in the working tree; the notebook rebuilds it from the
  Drive cache via `carve_splits.py` and asserts the final-eval lock).
- **Proposed, not owner-confirmed:** 30 steps x 4 prompts x 8 rollouts, lr 2e-5, `beta=0.02`,
  LoRA r16, sampling T=1.0 / top_p=1.0 (on-policy; the M07b pass was at top_p 0.95), 60-minute
  training cap, 128 train + 32 held-out requests. Cost/cap must be confirmed before running.
- **To run:** commit and push the code to the branch named in the notebook (`REF`), upload the
  frozen target detector folder to Drive, and have the AI4Privacy cache on Drive.

## Owner ideas for RL signal and control (2026-11-13, not decided, none implemented)

1. **Small SFT warm start on detector errors** so RL starts with signal: target detector misses ~1.4% of
   base-2B generations, so a G=8 group contains a miss only ~10% of the time. Constraints if
   done: the SFT texts must come from train-side data (never `short_eval_*`, `short_eval_neg_v1`
   or any comparison/dev request, whose base/target detector misses are evaluation data); the unmodified
   `SFT attacker` stays separate (plan M05 keeps it as an optional comparison arm); same prompt,
   schema and non-thinking config as RL. Decide after the smoke run measures the real
   miss rate per group; it is the fallback if the signal is too sparse.
2. **Fluency control** (readable English; RL could drift to odd text, or the detector must be
   robust to odd text): first control is `beta > 0` (D18); otherwise inspect samples during
   the smoke run and add a guard later only if drift shows.
3. **Diversity control** (not collapsing onto one detector weakness; lower priority):
   monitor duplicate/near-duplicate rate, length and opening-pattern share during the first
   real RL run; an API-model check or reward term only if collapse appears.

## Blockers / open decisions

- D01, D04, D06–D11 open (see `docs/DECISIONS.md`). D02/D03/D05/D12 settled per plan v1.3;
  split/name policies recorded as R05/R06; pseudo-name filter approved and implemented (R07).
  No training, no paid spend, no final-test use.
  M03.1/M05.1 gate closed (8/8 PASS on the 2026-10-04 re-run).
