#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b17_tpjepa_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"

PROJECT_ROOT=${PROJECT_ROOT}

cd "$PROJECT_ROOT"
test "$(git rev-parse --show-toplevel)" = "$PROJECT_ROOT"

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
export PYTHONPATH="$PROJECT_ROOT":${PYTHONPATH:-}
cd "$PROJECT_ROOT"
python -m py_compile \
  app/hdepic_lora_action_anticipation/task_protected_jepa_pruning.py \
  app/hdepic_lora_action_anticipation/export_task_protected_jepa_targets.py \
  app/hdepic_lora_action_anticipation/evaluate_task_protected_jepa_oracle.py \
  app/hdepic_lora_action_anticipation/evaluate_attention_jepa_swap_oracle.py \
  app/hdepic_lora_action_anticipation/evaluate_predictor_compensated_completion_e0.py \
  scripts/build_b17_tpjepa_oracle_manifest.py \
  scripts/analyze_b17_tpjepa_calibration_targets.py \
  scripts/analyze_b17_tpjepa_swap_oracle.py \
  scripts/analyze_b17_predictor_completion_e0.py \
  scripts/test_b17_task_protected_jepa.py
python scripts/test_b17_task_protected_jepa.py
EOF
