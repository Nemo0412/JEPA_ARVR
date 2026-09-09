#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_single_frame_full64
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
export Q3_SF_ROOT="${DATA_ROOT}/outputs/attn_corner_sink/q3_single_frame_source"
export Q3_SF_OUT="${Q3_SF_ROOT}/full64-viz-${SLURM_JOB_ID}"
export TMPDIR="${Q3_SF_ROOT}/full64-tmp-${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR}/matplotlib"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python -m py_compile scripts/egtea/render_b18_q3_single_frame_full64.py
python scripts/egtea/render_b18_q3_single_frame_full64.py \
 --manifest "${Q3_SF_ROOT}/prepared/manifest.json" \
 --source-analysis "${Q3_SF_ROOT}/analysis-17111020" --out "$Q3_SF_OUT"
EOF
