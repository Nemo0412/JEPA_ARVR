#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_token_self_review
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:03:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT}
export CODE_ROOT="${PROJECT_ROOT}"
export TS_REVIEW_INPUT="${DATA_ROOT}/outputs/attn_corner_sink/q3_token_self_return/analysis-17158362"
export TS_REVIEW_OUTPUT="${DATA_ROOT}/outputs/attn_corner_sink/q3_token_self_return/review-${SLURM_JOB_ID}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PROJECT_ROOT="$CODE_ROOT" SHARED_PROJECT_ROOT="$DATA_ROOT" PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python scripts/egtea/archive_b18_q3_token_self_review.py
EOF
