#!/usr/bin/env bash
# M06 baselines + matched M10 retraining, run on the RTX 3090 (ssh rtx3090), in order:
#   1. base-attacker candidate pool (configs/m06_base_generate.yaml, 600 requests x 4 samples)
#   2. plain_aug_v1 and gf_aug_v1 from that pool (scripts/finalize_base_aug_v1.py)
#   3. retrain the target detector on template, plain and gf (configs/m10_retrain_<arm>.yaml)
#   4. score every retrained detector, and the RL pilot's, on held-out RL letters and on
#      held-out base-attacker texts (scripts/score_rl_letters.py)
# Each step refuses to start at or after DEADLINE_HOUR (default 7) or when another
# pii_redteam job uses the GPU. Usage on the 3090:
#   setsid nohup bash scripts/run_m06_baselines_3090.sh > ~/runs/m06_baselines.log 2>&1 < /dev/null &
set -euo pipefail
cd "$HOME/pii-redteam-rl"
PY="$HOME/rl-env/bin/python"
RUNS="$HOME/runs"
DEADLINE_HOUR="${DEADLINE_HOUR:-7}"
TARGET=artifacts/d1-frozen-d1v2-20261110-075731
RL_PILOT="$RUNS/m10-retrain-rl-pilot-20261115-174807"
export PYTHONPATH="$PWD/src"
export PROBE_REPO_COMMIT="${PROBE_REPO_COMMIT:-7a8c733+uncommitted}"
set -a; . "$HOME/.config/pii-redteam/mistral.env"; set +a

step() {
  local hour
  hour=$((10#$(date +%H)))
  if (( hour >= DEADLINE_HOUR && hour < 12 )); then
    echo "STOP: $(date) is past the ${DEADLINE_HOUR}:00 deadline, not starting: $1"; exit 3
  fi
  if ! nvidia-smi > /dev/null 2>&1; then echo "STOP: nvidia-smi fails (GPU gone?)"; exit 4; fi
  echo "=== $(date '+%F %T') $1"
}

chosen_dir() {  # the epoch the freeze rule chose, from a train-detector run directory
  "$PY" -c "import json,sys; r=json.load(open(sys.argv[1]+'/d1_report.json')); \
print(sys.argv[1]+'/d1-epoch'+str(r['freeze_selection']['chosen_epoch']))" "$1"
}

step "generate base-attacker candidates"
"$PY" -m pii_redteam.cli rl-generate --config configs/m06_base_generate.yaml
GEN=$(ls -td "$RUNS"/m06-base-generate-* | head -1)

step "select plain and generate-and-filter sets from $GEN"
"$PY" scripts/finalize_base_aug_v1.py --run-dir "$GEN"
for arm in plain gf; do
  sha=$(sha256sum "data/synthetic/${arm}_aug_v1.jsonl" | cut -d' ' -f1)
  sed -i "s/SHA_${arm}\b/${sha}/" "configs/m10_retrain_${arm}.yaml"
done

for arm in template plain gf; do
  step "retrain the target detector on ${arm}_aug_v1"
  "$PY" -m pii_redteam.cli train-detector --config "configs/m10_retrain_${arm}.yaml"
done

for arm in template plain gf rl-pilot; do
  run=$(ls -td "$RUNS"/m10-retrain-"$arm"-* | head -1)
  after=$(chosen_dir "$run")
  step "score $after"
  "$PY" scripts/score_rl_letters.py --before "$TARGET" --after "$after" \
    --samples "$HOME/heldout/longrun/samples_post.jsonl" "$HOME/heldout/pilot3/samples_post.jsonl" \
    --out "$run/held_out_letters_m06.json"
  "$PY" scripts/score_rl_letters.py --before "$TARGET" --after "$after" \
    --samples "$HOME/heldout/base_pre/samples_pre.jsonl" --out "$run/held_out_base_texts.json"
done
echo "=== $(date '+%F %T') ALL DONE"
