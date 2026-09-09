#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_single_frame
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
export CODE_ROOT="${PROJECT_ROOT}" VJEPA_ROOT="${PROJECT_ROOT}/vjepa2"
export SF_START="${SF_START:-0}" SF_STOP="${SF_STOP:-2}" SF_PARITY="${SF_PARITY:-1}"
export SF_TAG="${SF_TAG:-b18-q3-single-frame-smoke}" SF_REFERENCE="${SF_REFERENCE:-}"
export SF_RUN="${DATA_ROOT}/outputs/attn_corner_sink/q3_single_frame_source/${SF_TAG}-${SLURM_JOB_ID}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$SF_RUN"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "${SF_RUN}/gpu_identity.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits --loop=5 > "${SF_RUN}/gpu.csv" &
SF_MONITOR_PID=$!
stop_monitor() { kill "$SF_MONITOR_PID" 2>/dev/null || true; wait "$SF_MONITOR_PID" 2>/dev/null || true; }
trap stop_monitor EXIT INT TERM
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PROJECT_ROOT="$CODE_ROOT" SHARED_PROJECT_ROOT="$DATA_ROOT" PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
extra=()
[[ "$SF_PARITY" == 1 ]] && extra+=(--parity)
[[ -n "$SF_REFERENCE" ]] && extra+=(--smoke-reference "$SF_REFERENCE")
python -m app.hdepic_lora_action_anticipation.probe_b18_q3_single_frame_source gpu \
 --manifest "${DATA_ROOT}/outputs/attn_corner_sink/q3_single_frame_source/prepared/manifest.json" \
 --archive-anchor "${DATA_ROOT}/outputs/attn_corner_sink/q3_oldest_mechanism/b18-q3-mechanism-full-a-17092450/capture/q3_000.npz" \
 --out "${SF_RUN}/capture" --start "$SF_START" --stop "$SF_STOP" --tag "$SF_TAG" "${extra[@]}"
EOF
