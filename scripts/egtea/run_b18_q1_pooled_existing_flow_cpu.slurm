#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q1_pooled_existing_flow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT}
export CODE_ROOT="${PROJECT_ROOT}"
export Q1_POOLED_OUT="${DATA_ROOT}/outputs/attn_corner_sink/q1_target_crossslot/pooled-existing-flow-${SLURM_JOB_ID}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python scripts/egtea/analyze_b18_q1_pooled_existing_flow.py \
  --source-dir "${DATA_ROOT}/outputs/attn_corner_sink/q1_paired_hybrid/b18-q1-paired-hybrid-full4000-17007365" \
  --out-dir "$Q1_POOLED_OUT"
EOF
