#!/bin/bash
#SBATCH --job-name=JEPA_ARVR__enc_d18_projector
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --time=00:45:00
#SBATCH --output=logs/enc_d18_projector_%j.out
#SBATCH --error=logs/enc_d18_projector_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"
RUN_CONFIG="${RUN_CONFIG:-${PROJECT_ROOT}/configs/examples/b14_encoder_d18_projector.yaml}"
PERF_INTERVAL="${PERF_INTERVAL:-30}"
PERF_PREFIX="${PROJECT_ROOT}/logs/perf_${SLURM_JOB_ID:-manual}_encoder_projector"

export PROJECT_ROOT SIF OVERLAY ENV_NAME RUN_CONFIG PERF_INTERVAL PERF_PREFIX
mkdir -p "${PROJECT_ROOT}/logs"

monitor_pids=()
stop_monitors() {
    for pid in "${monitor_pids[@]:-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        fi
    done
}
trap stop_monitors EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv -l "${PERF_INTERVAL}" > "${PERF_PREFIX}_nvidia_smi.csv" 2> "${PERF_PREFIX}_nvidia_smi.err" &
    monitor_pids+=("$!")
fi
if command -v vmstat >/dev/null 2>&1; then
    vmstat -t "${PERF_INTERVAL}" > "${PERF_PREFIX}_vmstat.log" 2> "${PERF_PREFIX}_vmstat.err" &
    monitor_pids+=("$!")
fi

singularity exec --nv --overlay "${OVERLAY}:ro" "${SIF}" /bin/bash -lc 'bash -s' <<'EOF'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate "/ext3/envs/${ENV_NAME}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
python scripts/train_encoder_exit_projector.py --run-config "${RUN_CONFIG}"
EOF

trap - EXIT
stop_monitors

if [[ -s "${PERF_PREFIX}_nvidia_smi.csv" ]]; then
    awk -F',' '
        NR > 1 {
            value=$3
            gsub(/[^0-9.]/, "", value)
            if (value != "") { sum += value; count += 1; if (value > max) max=value }
        }
        END {
            if (count > 0) printf("[perf-summary] gpu_util_samples=%d mean=%.2f%% max=%.2f%%\n", count, sum/count, max)
        }
    ' "${PERF_PREFIX}_nvidia_smi.csv"
fi
