#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_boundaries
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT}
export CODE_ROOT="${PROJECT_ROOT}"
export VJEPA_ROOT
export Q3_START="${Q3_START:-0}" Q3_STOP="${Q3_STOP:-2}" Q3_PARITY="${Q3_PARITY:-1}"
export Q3_TAG="${Q3_TAG:-b18-q3-boundary-smoke}"
export Q3_REFERENCE="${Q3_REFERENCE:-}"
export Q3_RUN="${DATA_ROOT}/outputs/attn_corner_sink/q3_boundaries/${Q3_TAG}-${SLURM_JOB_ID}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$Q3_RUN"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "${Q3_RUN}/gpu_identity.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits --loop=5 > "${Q3_RUN}/gpu.csv" &
Q3_MONITOR_PID=$!
stop_monitor() { kill "$Q3_MONITOR_PID" 2>/dev/null || true; wait "$Q3_MONITOR_PID" 2>/dev/null || true; }
trap stop_monitor EXIT INT TERM
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PROJECT_ROOT="$CODE_ROOT" SHARED_PROJECT_ROOT="$DATA_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
extra=()
[[ "$Q3_PARITY" == 1 ]] && extra+=(--parity)
[[ -n "$Q3_REFERENCE" ]] && extra+=(--reference "$Q3_REFERENCE")
python -m app.hdepic_lora_action_anticipation.probe_b18_q3_boundaries gpu \
 --manifest "${DATA_ROOT}/outputs/attn_corner_sink/q3_boundaries/prepared/manifest.json" \
 --out "${Q3_RUN}/capture" --start "$Q3_START" --stop "$Q3_STOP" --tag "$Q3_TAG" "${extra[@]}"
EOF
