#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_mechanism
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
export MECH_START="${MECH_START:-0}" MECH_STOP="${MECH_STOP:-2}" MECH_PARITY="${MECH_PARITY:-1}"
export MECH_TAG="${MECH_TAG:-b18-q3-mechanism-smoke}"
export MECH_RUN="${DATA_ROOT}/outputs/attn_corner_sink/q3_oldest_mechanism/${MECH_TAG}-${SLURM_JOB_ID}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$MECH_RUN"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "${MECH_RUN}/gpu_identity.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits --loop=5 > "${MECH_RUN}/gpu.csv" &
MECH_MONITOR_PID=$!
stop_monitor() { kill "$MECH_MONITOR_PID" 2>/dev/null || true; wait "$MECH_MONITOR_PID" 2>/dev/null || true; }
trap stop_monitor EXIT INT TERM
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PROJECT_ROOT="$CODE_ROOT" SHARED_PROJECT_ROOT="$DATA_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
extra=()
[[ "$MECH_PARITY" == 1 ]] && extra+=(--parity)
python -m app.hdepic_lora_action_anticipation.probe_b18_q3_oldest_mechanism gpu \
 --manifest "${DATA_ROOT}/outputs/attn_corner_sink/q3_oldest_mechanism/prepared/manifest.json" \
 --reference "${DATA_ROOT}/outputs/attn_corner_sink/q3_boundaries/b18-q3-boundary-full-a-17059417/capture/q3_000.npz" \
 --out "${MECH_RUN}/capture" --start "$MECH_START" --stop "$MECH_STOP" --tag "$MECH_TAG" "${extra[@]}"
EOF
