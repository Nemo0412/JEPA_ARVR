#!/bin/bash
# B13 CPU logic test for the route-3 gating probe math (spearman / top-K overlap /
# ridge accumulate->solve). No model / video / GPU. PROJECT_ROOT points at the
# worktree holding the new code so the real module is compiled + imported.
#SBATCH --job-name=VJEPA2-EXP__b13_probe_test
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
    app/hdepic_lora_action_anticipation/gating_probe_saliency.py \
    app/hdepic_lora_action_anticipation/eval_stream_mtp_kvcache_prune.py \
    scripts/test_b13_gating_probe.py
  python scripts/test_b13_gating_probe.py
EOF
