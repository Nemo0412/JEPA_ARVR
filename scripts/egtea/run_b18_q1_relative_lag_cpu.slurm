#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q1_relative_lag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT=${DATA_ROOT}
CODE_ROOT="${PROJECT_ROOT}"
Q1_LAG_SOURCE="${DATA_ROOT}/outputs/attn_corner_sink/q1_target_crossslot/b18-q1-target-crossslot-full4000-17026550"
Q1_LAG_OUT="${DATA_ROOT}/outputs/attn_corner_sink/q1_position_lag/relative-lag-${SLURM_JOB_ID}"
export DATA_ROOT CODE_ROOT Q1_LAG_SOURCE Q1_LAG_OUT
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python scripts/egtea/analyze_b18_q1_relative_lag.py --source-dir "${Q1_LAG_SOURCE}" --out-dir "${Q1_LAG_OUT}"
EOF
