#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_random_acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT}
export CODE_ROOT="${PROJECT_ROOT}"
export VJEPA_ROOT
export OUTROOT="${DATA_ROOT}/outputs/attn_corner_sink/q3_random_drop_accuracy"
: "${Q3_RANDOM_PREP:?provide immutable prep directory}"
export Q3_RANDOM_PREP Q3_RANDOM_START="${Q3_RANDOM_START:-0}" Q3_RANDOM_STOP="${Q3_RANDOM_STOP:-64}"
export Q3_RANDOM_BATCH="${Q3_RANDOM_BATCH:-4}" Q3_RANDOM_WORKERS="${Q3_RANDOM_WORKERS:-4}"
export Q3_RANDOM_TAG="${Q3_RANDOM_TAG:-b18-q3-random-smoke}" Q3_RANDOM_REFERENCE="${Q3_RANDOM_REFERENCE:-}"
export Q3_RANDOM_OUT="${OUTROOT}/${Q3_RANDOM_TAG}-${SLURM_JOB_ID}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$Q3_RANDOM_OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "${Q3_RANDOM_OUT}/gpu_identity.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits --loop=5 > "${Q3_RANDOM_OUT}/gpu.csv" &
MONITOR_PID=$!
stop_monitor() { kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
trap stop_monitor EXIT INT TERM
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
extra=(); [[ -n "$Q3_RANDOM_REFERENCE" ]] && extra+=(--reference "$Q3_RANDOM_REFERENCE")
python -m app.hdepic_lora_action_anticipation.eval_b18_q3_random_drop_accuracy eval \
 --manifest "${Q3_RANDOM_PREP}/manifest.json" --out "$Q3_RANDOM_OUT/eval" --tag "$Q3_RANDOM_TAG" \
 --start "$Q3_RANDOM_START" --stop "$Q3_RANDOM_STOP" --batch "$Q3_RANDOM_BATCH" --workers "$Q3_RANDOM_WORKERS" "${extra[@]}"
EOF
