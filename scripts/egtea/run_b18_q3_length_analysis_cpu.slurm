#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_length_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT}
export CODE_ROOT="${PROJECT_ROOT}"
export VJEPA_ROOT
export Q3_ROOT="${DATA_ROOT}/outputs/attn_corner_sink/q3_length_accuracy"
: "${Q3_PREP:?provide frozen prep directory}" "${Q3_RUNS:?provide space-separated immutable eval paths}"
export Q3_PREP Q3_RUNS
export Q3_PARTIAL="${Q3_PARTIAL:-0}" Q3_BOOTSTRAP="${Q3_BOOTSTRAP:-10000}"
export Q3_ANALYSIS_TAG="${Q3_ANALYSIS_TAG:-analysis-full}"
export Q3_FAILED_ANCHORS="${Q3_FAILED_ANCHORS:-}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
extra=()
[[ "$Q3_PARTIAL" == 1 ]] && extra+=(--partial)
if [[ -n "$Q3_FAILED_ANCHORS" ]]; then
 read -r -a failed_anchors <<< "$Q3_FAILED_ANCHORS"
 extra+=(--failed-anchors "${failed_anchors[@]}")
fi
read -r -a runs <<< "$Q3_RUNS"
python -m py_compile scripts/egtea/analyze_b18_q3_length_accuracy.py
python scripts/egtea/analyze_b18_q3_length_accuracy.py --manifest "${Q3_PREP}/manifest.json" \
 --runs "${runs[@]}" --out "${Q3_ROOT}/${Q3_ANALYSIS_TAG}-${SLURM_JOB_ID}" \
 --bootstrap "$Q3_BOOTSTRAP" "${extra[@]}"
EOF
