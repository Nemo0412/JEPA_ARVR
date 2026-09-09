#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_fixed4s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:05:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT}
export CODE_ROOT="${PROJECT_ROOT}"
export VJEPA_ROOT
export Q3_ROOT="${DATA_ROOT}/outputs/attn_corner_sink/q3_boundaries"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PROJECT_ROOT="$CODE_ROOT" SHARED_PROJECT_ROOT="$DATA_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python -m py_compile scripts/egtea/analyze_b18_q3_fixed4s.py
python scripts/egtea/analyze_b18_q3_fixed4s.py \
 --manifest "${Q3_ROOT}/prepared/manifest.json" \
 --source "${Q3_ROOT}/analysis-full86" \
 --out "${Q3_ROOT}/analysis-fixed4s-${SLURM_JOB_ID}"
EOF
