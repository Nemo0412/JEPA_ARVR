#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_biadapter_diag
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160GB
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_binary_input_adapter_diag_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_binary_input_adapter_diag_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_binary_input_adapter_radius64.yaml}"
RUN_TAG="${RUN_TAG:-hdepic-lora-binary-input-adapter-r64-h8-dlmap-multihead-bs2-w12}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation/action_anticipation_frozen/${RUN_TAG}}"
ADAPTER_CHECKPOINT="${ADAPTER_CHECKPOINT:-${RUN_DIR}/binary_input_adapter_latest.pt}"
DIAG_DIR="${DIAG_DIR:-${RUN_DIR}/diagnostics}"
MAX_BATCHES="${MAX_BATCHES:-16}"
MAX_WORKERS="${MAX_WORKERS:-4}"
WITH_MODEL="${WITH_MODEL:-1}"
FULL_TRAIN_SCAN="${FULL_TRAIN_SCAN:-1}"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export CONFIG_PATH RUN_TAG RUN_DIR ADAPTER_CHECKPOINT DIAG_DIR MAX_BATCHES MAX_WORKERS WITH_MODEL FULL_TRAIN_SCAN

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

COMMON_ARGS=(
  --config "${CONFIG_PATH}"
  --max-workers "${MAX_WORKERS}"
)

ADAPTER_ARGS=()
if [[ -f "${ADAPTER_CHECKPOINT}" ]]; then
  ADAPTER_ARGS+=(--adapter-checkpoint "${ADAPTER_CHECKPOINT}")
fi
if [[ "${WITH_MODEL}" == "1" ]]; then
  ADAPTER_ARGS+=(--with-model)
fi

python -m app.hdepic_lora_action_anticipation.binary_input_adapter_diagnostics \
  "${COMMON_ARGS[@]}" \
  "${ADAPTER_ARGS[@]}" \
  --max-batches "${MAX_BATCHES}" \
  --split val \
  --output "${DIAG_DIR}/val_adapter_space.json"

python -m app.hdepic_lora_action_anticipation.binary_input_adapter_diagnostics \
  "${COMMON_ARGS[@]}" \
  "${ADAPTER_ARGS[@]}" \
  --max-batches "${MAX_BATCHES}" \
  --split train \
  --output "${DIAG_DIR}/train_adapter_space.json"

if [[ "${FULL_TRAIN_SCAN}" == "1" ]]; then
  python -m app.hdepic_lora_action_anticipation.binary_input_adapter_diagnostics \
    "${COMMON_ARGS[@]}" \
    --max-batches 0 \
    --split train \
    --cpu \
    --output "${DIAG_DIR}/train_full_coverage.json"
fi
BASH
)

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
