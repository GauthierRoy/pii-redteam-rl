#!/usr/bin/env bash
# M10 round 2 on the RTX 3090 (ssh rtx3090): retrain the RL-retrained detector (round 1) on
# rl_aug_v2, gf_aug_v2 and plain_aug_v2 (configs/m10_round2_retrain_<arm>.yaml), then score
# the start detector and the three new ones on:
#   - the 1000-step person attack's held-out texts (182 pre-RL, 240 post-RL; 32 names never in
#     any augmentation set): ~/runs/heldout_cases_person_modal_20261124.jsonl
#   - the name-swap probe of 2026-11-22 (bare name first, common names swapped in)
#   - round 1's held-out sets (RL letters, base-attacker texts) and the name-free messages
#     (scripts/score_rl_letters.py), so a regression on round 1's weakness shows up.
# One job at a time; stops if nvidia-smi fails. Usage on the 3090:
#   setsid nohup bash scripts/run_round2_3090.sh > ~/runs/m10_round2.log 2>&1 < /dev/null &
set -euo pipefail
cd "$HOME/pii-redteam-rl"
PY="$HOME/rl-env/bin/python"
RUNS="$HOME/runs"
START="$RUNS/m10-retrain-rl-pilot-20261115-174807/d1-epoch3"
OUT="$RUNS/m10-round2-scores"
export PYTHONPATH="$PWD/src"
export PROBE_REPO_COMMIT="${PROBE_REPO_COMMIT:-unrecorded}"
mkdir -p "$OUT"

step() {
  if ! nvidia-smi > /dev/null 2>&1; then echo "STOP: nvidia-smi fails (GPU gone?)"; exit 4; fi
  echo "=== $(date '+%F %T') $1"
}

chosen_dir() {  # the epoch the freeze rule chose, from a train-detector run directory
  "$PY" -c "import json,sys; r=json.load(open(sys.argv[1]+'/d1_report.json')); \
print(sys.argv[1]+'/d1-epoch'+str(r['freeze_selection']['chosen_epoch']))" "$1"
}

for arm in rl gf plain; do
  sha=$(sha256sum "data/synthetic/${arm}_aug_v2.jsonl" | cut -d' ' -f1)
  sed -i "s/SHA_${arm}\b/${sha}/" "configs/m10_round2_retrain_${arm}.yaml"
  if ls -d "$RUNS"/m10-round2-retrain-"$arm"-* > /dev/null 2>&1; then
    echo "skip $arm: already trained"; continue
  fi
  step "retrain on ${arm}_aug_v2"
  "$PY" -m pii_redteam.cli train-detector --config "configs/m10_round2_retrain_${arm}.yaml"
done

DETECTORS=(--detector "start=$START")
for arm in rl gf plain; do
  run=$(ls -td "$RUNS"/m10-round2-retrain-"$arm"-* | head -1)
  DETECTORS+=(--detector "${arm}2=$(chosen_dir "$run")")
done

step "held-out attack texts and name-swap probe (CPU)"
CUDA_VISIBLE_DEVICES= "$PY" scripts/probe_name_swap.py "$RUNS/heldout_cases_person_modal_20261124.jsonl" \
  "${DETECTORS[@]}" --out "$OUT/heldout_person_attack.json"
CUDA_VISIBLE_DEVICES= "$PY" scripts/probe_name_swap.py "$RUNS/name_cases_20261122.jsonl" \
  "${DETECTORS[@]}" --out "$OUT/name_swap.json"

for arm in rl gf plain; do
  run=$(ls -td "$RUNS"/m10-round2-retrain-"$arm"-* | head -1)
  after=$(chosen_dir "$run")
  step "round-1 held-out sets for $after"
  "$PY" scripts/score_rl_letters.py --before "$START" --after "$after" \
    --samples "$HOME/heldout/longrun/samples_post.jsonl" "$HOME/heldout/pilot3/samples_post.jsonl" \
    --out "$run/held_out_letters_round1.json"
  "$PY" scripts/score_rl_letters.py --before "$START" --after "$after" \
    --samples "$HOME/heldout/base_pre/samples_pre.jsonl" --out "$run/held_out_base_texts.json"
done
echo "=== $(date '+%F %T') ALL DONE"
