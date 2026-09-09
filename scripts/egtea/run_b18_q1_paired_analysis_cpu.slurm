#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q1_paired_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT=${DATA_ROOT}
CODE_ROOT="${PROJECT_ROOT}"
: "${Q1_SHARDS:?Provide space-separated completed shard directories}"
: "${Q1_ANALYSIS_OUT:?Provide fresh analysis output directory}"
Q1_ALLOW_PARTIAL="${Q1_ALLOW_PARTIAL:-0}"
Q1_BOOTSTRAP_REPS="${Q1_BOOTSTRAP_REPS:-9999}"
Q1_PRELUDE_SHARD="${Q1_PRELUDE_SHARD:-}"
Q1_PRELUDE_OUT="${Q1_PRELUDE_OUT:-}"
export DATA_ROOT CODE_ROOT Q1_SHARDS Q1_ANALYSIS_OUT Q1_ALLOW_PARTIAL Q1_BOOTSTRAP_REPS
export Q1_PRELUDE_SHARD Q1_PRELUDE_OUT
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
read -r -a shards <<< "${Q1_SHARDS}"
extra=()
[[ "${Q1_ALLOW_PARTIAL}" == "1" ]] && extra+=(--allow-partial)
if [[ -n "${Q1_PRELUDE_SHARD}" ]]; then
  : "${Q1_PRELUDE_OUT:?A prelude requires a fresh output directory}"
  python scripts/egtea/analyze_b18_q1_paired_hybrid.py --shards "${Q1_PRELUDE_SHARD}" \
    --out-dir "${Q1_PRELUDE_OUT}" --allow-partial --bootstrap-reps 199 --no-plots
fi
python scripts/egtea/analyze_b18_q1_paired_hybrid.py --shards "${shards[@]}" \
  --out-dir "${Q1_ANALYSIS_OUT}" --bootstrap-reps "${Q1_BOOTSTRAP_REPS}" "${extra[@]}"
EOF
