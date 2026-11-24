#!/usr/bin/env bash
# LLM-writer arm, board v2, 200 steps per writer, one after another on the 3090 PC (CPU only;
# the writers share one Mistral key, so never in parallel). Usage on the PC:
#   setsid nohup bash scripts/run_llm_v2_3090.sh > ~/runs/m10_llm_v2.log 2>&1 < /dev/null &
set -uo pipefail
cd "$HOME/pii-redteam-rl"
export PYTHONPATH="$PWD/src"
export PROBE_REPO_COMMIT="${PROBE_REPO_COMMIT:-unrecorded}"
set -a; . "$HOME/.config/pii-redteam/mistral.env"; set +a
for writer in ${WRITERS:-large deepseek glm}; do
  echo "=== $(date '+%F %T') start $writer"
  "$HOME/rl-env/bin/python" -m pii_redteam.cli llm-attack \
    --config "configs/m10_llm_attack_v2_${writer}_3090.yaml" || echo "=== $writer FAILED (exit $?)"
done
echo "=== $(date '+%F %T') ALL DONE"
