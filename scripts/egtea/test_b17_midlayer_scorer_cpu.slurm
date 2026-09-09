#!/bin/bash
# B17 CPU logic test for the learned-early-pruner scorer module (MLP + _prune math).
# No encoder/GPU. PROJECT_ROOT points at the worktree holding the new code.
#SBATCH --job-name=VJEPA2-EXP__b17_scorer_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:10:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT}}"
export VJEPA_ROOT
export PROJECT_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
  set -euo pipefail
  export PYTHONNOUSERSITE=1
  export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
  cd "${PROJECT_ROOT}"
  python -m py_compile \
    app/hdepic_lora_action_anticipation/midlayer_scorer.py \
    app/hdepic_lora_action_anticipation/distill_midlayer_scorer.py \
    app/hdepic_lora_action_anticipation/train_stream_mtp_learned_prune.py \
    scripts/test_b17_midlayer_scorer.py
  python scripts/test_b17_midlayer_scorer.py
EOF
