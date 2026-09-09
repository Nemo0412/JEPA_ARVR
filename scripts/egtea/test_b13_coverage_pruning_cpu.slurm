#!/bin/bash
# B13 CPU logic test for the Direction-A coverage / de-redundancy prune selector.
# Compiles + runs the isolated index-math test (no model / video / GPU).
# Point PROJECT_ROOT at the worktree holding the new code.
#SBATCH --job-name=VJEPA2-EXP__b13_cov_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:05:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
PROJECT_ROOT="${PROJECT_ROOT:-${DATA_ROOT}}"
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
    app/hdepic_lora_action_anticipation/coverage_pruning.py \
    app/hdepic_lora_action_anticipation/eval_stream_mtp_kvcache_prune.py \
    scripts/test_b13_coverage_pruning.py
  python scripts/test_b13_coverage_pruning.py
EOF
