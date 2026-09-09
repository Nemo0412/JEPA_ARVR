#!/bin/bash
# CPU-only semantic contracts for B17 fixed-budget gaze spatial pruning.
#SBATCH --job-name=VJEPA2-EXP__b17_gaze_prune_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT}}"
EXPECTED_CODE_ROOT="${PROJECT_ROOT}"
CODE_ROOT="${CODE_ROOT:-${EXPECTED_CODE_ROOT}}"
export VJEPA_ROOT
[[ "$(realpath "${CODE_ROOT}")" == "${EXPECTED_CODE_ROOT}" ]] \
  || { echo "ERROR: CPU contract must use shared code root" >&2; exit 2; }
[[ "$(realpath "$CODE_ROOT")" == "$(realpath "$PROJECT_ROOT")" ]] \
  || { echo "ERROR: wrong shared code root" >&2; exit 2; }
mkdir -p "${DATA_ROOT}/outputs/b17_token_pruning"
export CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python scripts/test_b17_gaze_spatial_pruning.py
EOF
