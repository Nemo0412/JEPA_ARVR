#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_length_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT}
export CODE_ROOT="${PROJECT_ROOT}"
export VJEPA_ROOT
export Q3_ROOT="${DATA_ROOT}/outputs/attn_corner_sink/q3_length_accuracy"
export Q3_MODE="${Q3_MODE:-prepare}" Q3_WORKERS="${Q3_WORKERS:-4}" Q3_PREP="${Q3_PREP:-${Q3_ROOT}/prepared-${SLURM_JOB_ID}}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python -m py_compile app/hdepic_lora_action_anticipation/eval_b18_q3_length_accuracy.py
if [[ "$Q3_MODE" == prepare ]]; then
 python -m app.hdepic_lora_action_anticipation.eval_b18_q3_length_accuracy prepare --out "$Q3_PREP" --cache-stop 32 --workers "$Q3_WORKERS"
elif [[ "$Q3_MODE" == cache ]]; then
 python -m app.hdepic_lora_action_anticipation.eval_b18_q3_length_accuracy cache --manifest "${Q3_PREP}/manifest.json" --workers "$Q3_WORKERS" --report "${Q3_ROOT}/cache-${SLURM_JOB_ID}.json"
else
 echo "unsupported mode $Q3_MODE" >&2
 exit 2
fi
EOF
