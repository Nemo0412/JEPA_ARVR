#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q1_target_crossslot_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT=${DATA_ROOT}
CODE_ROOT="${PROJECT_ROOT}"
: "${Q1_RUNS:?Provide completed run directory paths separated by spaces}"
: "${Q1_ANALYSIS_OUT:?Provide a fresh output directory}"
Q1_ALLOW_PARTIAL="${Q1_ALLOW_PARTIAL:-0}"
Q1_BOOTSTRAP_REPS="${Q1_BOOTSTRAP_REPS:-9999}"
export DATA_ROOT CODE_ROOT Q1_RUNS Q1_ANALYSIS_OUT Q1_ALLOW_PARTIAL Q1_BOOTSTRAP_REPS
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
read -r -a runs <<< "${Q1_RUNS}"
extra=()
[[ "${Q1_ALLOW_PARTIAL}" == "1" ]] && extra+=(--allow-partial)
python scripts/egtea/analyze_b18_q1_target_crossslot.py --runs "${runs[@]}" \
  --out-dir "${Q1_ANALYSIS_OUT}" --bootstrap-reps "${Q1_BOOTSTRAP_REPS}" "${extra[@]}"
EOF
