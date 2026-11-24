# pii-redteam-rl — RL-guided synthetic data for robust PII detection

A small LLM attacker (Qwen3.5-2B + LoRA, trained with GRPO) writes short messages that contain a
requested person's name; a frozen ModernBERT name detector scores them, and the attacker is
rewarded when the detector misses the name. The texts it learns to write become augmentation
data for retraining the detector.

**Primary question:** does RL-generated augmentation improve held-out person-name detection more
than ordinary generation, templates and generate-and-filter, at a matched budget?
Secondary: name disparities (M11); engineering: generation cost (plan S00–S05).

Read first:
- `PROJECT_PLAN.md` (plan v1.3, milestones M00–M13), `AGENT_START_HERE.md`
- `STATE.md` (session log, newest sections first), `EXPERIMENTS.yaml` (every run, with metrics)
- `docs/DECISIONS.md` (D01–D25), `docs/LIMITATIONS.md` (read before quoting any result; L1: the
  detectors were trained with B/I tags, read as PERSON / not-PERSON since D25)

## Status (2026-11-24)

All results are pilots: one seed, development sets only. The final evaluation set is locked
and has not been used.

| Milestone | State |
|---|---|
| M01–M05 | scaffold, data splits, base detector, request banks, attacker prompt (done) |
| M07 | target detector frozen (`d1-frozen-d1v2-20261110-075731`) |
| M08 | RL attacker on Modal (TRL GRPO + colocated vLLM); exploits found and gated (D18–D25) |
| M09 | augmentation sets from the RL attacker (`rl_aug_v1`, `rl_aug_v2`) |
| M06 / M10 | baselines matched to the RL sets (templates, plain generation, generate-and-filter) and retraining comparisons: rounds 1 and 2 done; LLM-writer arm in progress |
| M11–M13 | disparities, final test, write-up: not started |

What the pilots show so far (details and numbers in `EXPERIMENTS.yaml`):

- **Round 1** (against the target detector): RL found "Dear …" letters, which the detector
  missed 72% of the time. Retraining on RL texts closed it (0%), but generate-and-filter nearly
  matched (1.7%): base-model text already contains that pattern.
- **Round 2** (against the RL-retrained detector, 1000-step attack): RL found names as the first
  word followed by certain continuations ("Bob, hi there! …"); held-out misses 0/182 → 29/240,
  transferring to every detector. Retraining on RL texts closes it (29 → 0) where
  generate-and-filter (8) and plain generation (11) do not; the base model writes name-first
  texts but almost never ones the detector misses (0.3% vs 10.7%). Cost: RL's narrow set adds
  boundary errors and false positives on capitalised first words.
- **LLM-writer arm** (a large API model with the detector's feedback, same requests, filters
  and reward): GLM 5.3 with a small board found little (50 steps); Mistral Large with the large
  board v2 reached 25/194 held-out misses in 200 steps, on par with RL, with a different pattern
  (terse bureaucratic fragments). DeepSeek and GLM v2 runs in progress.

## Quickstart

```bash
uv sync                          # dependencies (creates .venv + uv.lock)
make test                        # unit + integration tests, CPU-only
bash scripts/lint.sh             # ruff + ty
```

Every experiment is one CLI command on a YAML config (`PYTHONPATH=src python -m pii_redteam.cli`):

| Command | What it does |
|---|---|
| `rl-smoke --config C` | GRPO attack run (the RL arm; any length, despite the name) |
| `rl-generate --config C` | sample an attacker (RL adapter or the base model) on fresh names, filter, write accepted texts |
| `train-detector --config C` | fine-tune a frozen detector on an augmentation set, score dev and short_eval |
| `llm-attack --config C` | LLM-writer arm: an API model writes texts with the detector's feedback |
| `probe-base`, `compare-prompts` | M07a/M07b probes |

Where runs go:
- **Modal** (L4, budget-capped): `scripts/modal_rl.py`. Always launch with
  `modal run --detach scripts/modal_rl.py::<pilot|long_pilot|generate> --config C`; runs are
  spawned, so they do not depend on the laptop staying awake. Results:
  `modal run scripts/modal_rl.py::result --call-id fc-...`.
- **RTX 3090** (a shared PC, `ssh rtx3090`, `*_3090.yaml` configs): detector retraining, CPU
  scoring and the LLM-writer runs. The card has dropped off the bus under long full load
  (overheating memory suspected); `gpu_cooldown` in a config pauses between hot steps.
  Scripts: `scripts/run_round2_3090.sh`, `scripts/run_llm_v2_3090.sh`.
- **API keys**: the person judge, audit and LLM writers read `MISTRAL_API_KEY` from the
  environment (never from the repo). The LLM writers (GLM, DeepSeek, Mistral Large) are reached
  through the same Mistral endpoint.

## Layout

```text
configs/                 one YAML per run (m08_* RL, m09/m10_* generation, retraining, LLM arm)
data/synthetic/          augmentation and evaluation sets with manifests (*_aug_v1/v2, short_eval_*)
src/pii_redteam/
  cli.py                 entry points (table above)
  requests.py prompt_v2.py validity_v2.py   request bank, attacker prompt, validity rules
  probe.py probe_metrics.py                  detector scoring (PERSON / not-PERSON reading)
  train_detector.py detector_metrics.py      retraining and span metrics
  baselines.py           template / plain / generate-and-filter selection
  data/                  AI4Privacy loading and split carving
  training/
    rl_smoke.py          GRPO run (TRL), checkpoints, pre/post held-out samples
    rl_reward.py         reward: validity, person judge, audit, opening guard, log-odds scale
    person_judge.py      Mistral judge + audit with backoff
    rl_generate.py       augmentation sampling and acceptance
    llm_attack.py        LLM-writer arm: board v1/v2, token and price meter, pacing, spend cap
    gpu_cooldown.py      pause training while the 3090 runs hot
scripts/                 finalize_* (build datasets), score_* / probe_* (evaluation), modal_rl.py,
                         run_*_3090.sh (chained jobs on the 3090)
tests/unit tests/integration   stdlib unittest, CPU-only
docs/                    decisions, limitations, data card, glossary, results notes
```

## Hardware and budget

- Modal: about $30 per month, spend tracked per run in `EXPERIMENTS.yaml`
  (an RL run of 1000 steps on an L4 is about $3.9).
- Mistral API: judge, audit and LLM writers. `llm-attack` counts tokens per model and stops at
  `api.max_usd`; its prices are list prices from the config, not verified.
- RTX 3090: free when available; one job at a time, nothing at night unless agreed.
