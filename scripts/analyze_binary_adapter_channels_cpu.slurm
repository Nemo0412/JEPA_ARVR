#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__binary_adapter_channels
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=00:10:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/binary_adapter_channels_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/binary_adapter_channels_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Diagnose how much the binary-gaze input channel was actually used by an
# already-trained BinaryMapInputAdapter checkpoint. Reads only the saved
# state_dict; no GPU, no dataset, no training.
#
# Pass the checkpoint path via CHECKPOINT env var or as $1:
#   sbatch --export=ALL,CHECKPOINT=/path/to/binary_input_adapter_latest.pt \
#          scripts/analyze_binary_adapter_channels_cpu.slurm
# or interactively:
#   bash scripts/analyze_binary_adapter_channels_cpu.slurm /path/to/ckpt.pt

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

DEFAULT_CKPT="${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation/action_anticipation_frozen/hdepic-lora-binary-input-adapter-r64-h8-dlmap-multihead-bs2-w12-gazefixed-gradisolate/binary_input_adapter_latest.pt"
CHECKPOINT="${CHECKPOINT:-${1:-${DEFAULT_CKPT}}}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[adapter-channels] Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs"

export PROJECT_ROOT SIF OVERLAY ENV_NAME CHECKPOINT

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}

python "${PROJECT_ROOT}/scripts/analyze_binary_adapter_channels.py" "${CHECKPOINT}"
BASH
)

singularity exec \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
