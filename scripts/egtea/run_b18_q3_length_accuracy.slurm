#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_length_accuracy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT}
export CODE_ROOT="${PROJECT_ROOT}"
export VJEPA_ROOT
export Q3_ROOT="${DATA_ROOT}/outputs/attn_corner_sink/q3_length_accuracy"
: "${Q3_PREP:?must provide immutable prep directory}"
export Q3_PREP
export Q3_START="${Q3_START:-0}" Q3_STOP="${Q3_STOP:-32}" Q3_BATCH="${Q3_BATCH:-2}" Q3_WORKERS="${Q3_WORKERS:-2}"
export Q3_TAG="${Q3_TAG:-b18-q3-length-smoke}" Q3_PARITY="${Q3_PARITY:-1}" Q3_REFERENCE="${Q3_REFERENCE:-}"
export Q3_OUT="${Q3_ROOT}/${Q3_TAG}-${SLURM_JOB_ID}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "${Q3_OUT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "${Q3_OUT}/gpu_identity.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits --loop=5 > "${Q3_OUT}/gpu.csv" &
Q3_MONITOR_PID=$!
stop_monitor() { kill "$Q3_MONITOR_PID" 2>/dev/null || true; wait "$Q3_MONITOR_PID" 2>/dev/null || true; }
trap stop_monitor EXIT INT TERM
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
extra=()
[[ "$Q3_PARITY" == 1 ]] && extra+=(--parity)
[[ -n "$Q3_REFERENCE" ]] && extra+=(--reference "$Q3_REFERENCE")
python -m app.hdepic_lora_action_anticipation.eval_b18_q3_length_accuracy eval \
 --manifest "${Q3_PREP}/manifest.json" --out "${Q3_OUT}/eval" --tag "$Q3_TAG" \
 --start "$Q3_START" --stop "$Q3_STOP" --batch "$Q3_BATCH" --workers "$Q3_WORKERS" "${extra[@]}"
EOF
