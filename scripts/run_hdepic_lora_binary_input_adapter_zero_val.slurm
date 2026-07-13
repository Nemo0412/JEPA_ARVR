#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__binary_adapter_zero_val
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_binary_adapter_zero_val_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_binary_adapter_zero_val_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Val-only control for the binary-input-adapter branch:
# load the trained classifiers from latest.pt, load the trained input adapter
# from binary_input_adapter_latest.pt, but force the emitted gaze channel to 0.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_TAG="${RUN_TAG:-hdepic-lora-binary-input-adapter-r64-h8-dlmap-multihead-bs2-w12}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation/action_anticipation_frozen/${RUN_TAG}}"
ZERO_VAL_TAG="${ZERO_VAL_TAG:-${RUN_TAG}-zero-channel-val}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"
ZERO_RUN_DIR="${OUTPUT_DIR}/action_anticipation_frozen/${ZERO_VAL_TAG}"

if [[ ! -f "${RUN_DIR}/latest.pt" ]]; then
    echo "[zero-val] Missing classifier checkpoint: ${RUN_DIR}/latest.pt" >&2
    exit 1
fi
if [[ ! -f "${RUN_DIR}/binary_input_adapter_latest.pt" ]]; then
    echo "[zero-val] Missing adapter checkpoint: ${RUN_DIR}/binary_input_adapter_latest.pt" >&2
    exit 1
fi

mkdir -p "${ZERO_RUN_DIR}"
cp -p "${RUN_DIR}/latest.pt" "${ZERO_RUN_DIR}/latest.pt"

export PROJECT_ROOT
export GAZE_MODE=binary_input_adapter
export OUTPUT_DIR
export LORA_TAG="${ZERO_VAL_TAG}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_binary_input_adapter_zero_val.yaml}"
export RESUME_CHECKPOINT=1
export VAL_ONLY=1
export BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP=1
export BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT="${BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT:-${RUN_DIR}/binary_input_adapter_latest.pt}"

exec bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
