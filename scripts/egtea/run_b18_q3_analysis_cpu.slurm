#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT}
export CODE_ROOT="${PROJECT_ROOT}"
export VJEPA_ROOT
export Q3_INPUTS="${Q3_INPUTS:?space-separated exact capture directories required}"
export Q3_ANALYSIS="${Q3_ANALYSIS:?unique output directory required}"
export Q3_PARTIAL="${Q3_PARTIAL:-0}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PROJECT_ROOT="$CODE_ROOT" SHARED_PROJECT_ROOT="$DATA_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
read -r -a q3_inputs <<< "$Q3_INPUTS"
extra=()
[[ "$Q3_PARTIAL" == 1 ]] && extra+=(--allow-partial)
python -m py_compile scripts/egtea/analyze_b18_q3_boundaries.py
python scripts/egtea/analyze_b18_q3_boundaries.py \
 --manifest "${DATA_ROOT}/outputs/attn_corner_sink/q3_boundaries/prepared/manifest.json" \
 --runs "${q3_inputs[@]}" --out "$Q3_ANALYSIS" "${extra[@]}"
EOF
