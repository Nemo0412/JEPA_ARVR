#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_onset_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --gres=gpu:1
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT}
export CODE_ROOT="${PROJECT_ROOT}"
export ONSET_ROOT="${DATA_ROOT}/outputs/attn_corner_sink/q3_layer_attention_onset"
export ONSET_MANIFEST="${ONSET_MANIFEST:-${ONSET_ROOT}/prepared-v2/manifest.json}"
export TMPDIR="${ONSET_ROOT}/tmp-${SLURM_JOB_ID}" MPLCONFIGDIR="${ONSET_ROOT}/mpl-${SLURM_JOB_ID}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
export ONSET_START="${ONSET_START:-0}" ONSET_STOP="${ONSET_STOP:-2}" ONSET_SMOKE="${ONSET_SMOKE:-1}"
export ONSET_TAG="${ONSET_TAG:-b18-q3-onset-smoke}" ONSET_ANCHORS="${ONSET_ANCHORS:-}"
export ONSET_RUN="${ONSET_ROOT}/${ONSET_TAG}-${SLURM_JOB_ID}"
mkdir -p "$ONSET_RUN"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "${ONSET_RUN}/gpu_identity.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits --loop=5 > "${ONSET_RUN}/gpu.csv" &
ONSET_MONITOR_PID=$!
stop_monitor() { kill "$ONSET_MONITOR_PID" 2>/dev/null || true; wait "$ONSET_MONITOR_PID" 2>/dev/null || true; }
trap stop_monitor EXIT INT TERM
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PROJECT_ROOT="$CODE_ROOT" SHARED_PROJECT_ROOT="$DATA_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
extra=(); [[ "$ONSET_SMOKE" == 1 ]] && extra+=(--smoke)
[[ -n "$ONSET_ANCHORS" ]] && extra+=(--anchors "$ONSET_ANCHORS")
python -m app.hdepic_lora_action_anticipation.probe_b18_q3_layer_attention_onset gpu \
 --manifest "$ONSET_MANIFEST" --out "${ONSET_RUN}/capture" --start "$ONSET_START" --stop "$ONSET_STOP" \
 --tag "$ONSET_TAG" "${extra[@]}"
EOF
