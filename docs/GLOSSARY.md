# Model names

Two model families. Decision IDs (`D01`–`D18` in `docs/DECISIONS.md`) are unrelated to either.

## Detectors (ModernBERT, PERSON token classification)

| Name | What it is | Old name |
|---|---|---|
| **base detector** | Trained on AI4Privacy only (M03). | D0 |
| **target detector** | Base detector fine-tuned on `target_detector_aug_v1` short messages (M07b). Frozen. The RL reward queries this one. | D1 |
| **retrained detector** | Trained from the target detector's recipe on each method's data, for the final comparison (M10). | D2 |

In code: `base_detector`, `target_detector`, `retrained_detector`. Code that works with any
detector says `detector` (`_score_with_detector`, `DetectorScorer`).

## Attackers (Qwen3.5-2B, writes messages containing a requested name)

| Name | What it is | Old name |
|---|---|---|
| **base attacker** | Untuned Qwen3.5-2B. RL starts from it. | base generator, base policy |
| **SFT attacker** | Fine-tuned on entity-to-text pairs (M05). Optional comparison arm. | G_sft, generator |
| **RL attacker** | Base attacker + LoRA trained with GRPO against the target detector (M08). | G_rl, policy |

"On-policy" and "behavior policy" keep their usual RL meaning. "Generator" now only means a
script that produces files (the `scripts/sft_v2_gen/` template generator, the notebook builder).

## Old names that remain on purpose

The rename on 2026-11-14 covered everything inside the repo. These names point at things
outside it and were left as they are:

| Name | What it is |
|---|---|
| `d0-20261004-142714`, `d0-best` | Base detector training run and checkpoint directory (Drive). |
| `d1-20261108-145253`, `d1-epoch2`, `d1-epoch3` | Target detector training run and checkpoints. |
| `d1-frozen-d1v2-20261110-075731` | Frozen target detector archive (Drive, Modal volume). |
| `d0-negative-fp-audit` | Run id of the base detector false-positive audit. |
| `feat/m07b-freeze-d1`, `data/d1-aug-deepseek-v1`, `fix/land-d1-aug-v1` | Git branch names. |

Reports and logs written before the rename (on Drive and the Modal volume) use the old field
names: `d0_truncated_target`, `accepted_for_d1`, `rewards/d1_reward/*`.

`target_detector_aug_v1` was `d1_aug_v1`. Only the row-id prefix changed; its manifest records
the old name and sha256 under `renamed_from`.
