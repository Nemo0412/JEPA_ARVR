#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_token_self
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT}
export CODE_ROOT="${PROJECT_ROOT}" VJEPA_ROOT="${PROJECT_ROOT}/vjepa2"
export TS_START="${TS_START:-0}" TS_STOP="${TS_STOP:-2}" TS_PARITY="${TS_PARITY:-1}" TS_BENCHMARK="${TS_BENCHMARK:-30}"
export TS_TAG="${TS_TAG:-b18-q3-token-self-smoke}" TS_REFERENCE="${TS_REFERENCE:-}"
export TS_RUN="${DATA_ROOT}/outputs/attn_corner_sink/q3_token_self_return/${TS_TAG}-${SLURM_JOB_ID}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TS_RUN"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "${TS_RUN}/gpu_identity.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits --loop=5 > "${TS_RUN}/gpu.csv" &
TS_MONITOR_PID=$!
stop_monitor() { kill "$TS_MONITOR_PID" 2>/dev/null || true; wait "$TS_MONITOR_PID" 2>/dev/null || true; }
trap stop_monitor EXIT INT TERM
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PROJECT_ROOT="$CODE_ROOT" SHARED_PROJECT_ROOT="$DATA_ROOT" PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
extra=()
[[ "$TS_PARITY" == 1 ]] && extra+=(--parity)
[[ -n "$TS_REFERENCE" ]] && extra+=(--smoke-reference "$TS_REFERENCE")
python -m app.hdepic_lora_action_anticipation.probe_b18_q3_token_self_return gpu \
 --manifest "${DATA_ROOT}/outputs/attn_corner_sink/q3_token_self_return/prepared-v2/manifest.json" \
 --out "${TS_RUN}/capture" --start "$TS_START" --stop "$TS_STOP" --tag "$TS_TAG" \
 --benchmark-seconds "$TS_BENCHMARK" "${extra[@]}"
EOF
