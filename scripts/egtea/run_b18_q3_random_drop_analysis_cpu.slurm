#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_random_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT} CODE_ROOT=${PROJECT_ROOT}
export VJEPA_ROOT
: "${Q3_RANDOM_PREP:?provide prep}" "${Q3_RANDOM_RUNS:?provide space-separated eval paths}"
export Q3_RANDOM_PREP Q3_RANDOM_RUNS Q3_RANDOM_PARTIAL="${Q3_RANDOM_PARTIAL:-0}"
export Q3_RANDOM_ANALYSIS_TAG="${Q3_RANDOM_ANALYSIS_TAG:-analysis-full}" Q3_RANDOM_BOOTSTRAP="${Q3_RANDOM_BOOTSTRAP:-10000}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"; export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
read -r -a runs <<< "$Q3_RANDOM_RUNS"; extra=(); [[ "$Q3_RANDOM_PARTIAL" == 1 ]] && extra+=(--partial)
python scripts/egtea/analyze_b18_q3_random_drop_accuracy.py --manifest "${Q3_RANDOM_PREP}/manifest.json" \
 --runs "${runs[@]}" --out "${OUTROOT}/${Q3_RANDOM_ANALYSIS_TAG}-${SLURM_JOB_ID}" --bootstrap "$Q3_RANDOM_BOOTSTRAP" "${extra[@]}"
EOF
