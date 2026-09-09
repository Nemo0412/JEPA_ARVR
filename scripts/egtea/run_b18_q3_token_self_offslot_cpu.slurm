#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_offslot_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT} CODE_ROOT=${PROJECT_ROOT}
export VJEPA_ROOT
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"; export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python -m py_compile scripts/egtea/archive_b18_q3_token_self_offslot.py
python scripts/egtea/archive_b18_q3_token_self_offslot.py \
 --source "${DATA_ROOT}/outputs/attn_corner_sink/q3_token_self_return/analysis-17158362" \
 --out "${DATA_ROOT}/outputs/attn_corner_sink/q3_token_self_return/offslot-review-${SLURM_JOB_ID}"
EOF
