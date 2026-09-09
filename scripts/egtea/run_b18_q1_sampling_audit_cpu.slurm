#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q1_sampling_audit
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
OUT_DIR="${DATA_ROOT}/outputs/attn_corner_sink/q1_evidence_audit_20260905"
export DATA_ROOT CODE_ROOT OUT_DIR PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
cd "${CODE_ROOT}"
python scripts/egtea/audit_b18_q1_sampling.py --data-root "${DATA_ROOT}" --out-dir "${OUT_DIR}"
EOF
