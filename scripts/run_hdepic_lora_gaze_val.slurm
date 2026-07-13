#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_gaze
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_gaze_val_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_gaze_val_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Val-only inference on a gaze-enhanced V-JEPA video tree. This intentionally
# reuses the existing LoRA checkpoint/tag and changes only the video root.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIGMA="${SIGMA:-96}"
GAZE_VIDEO_ROOT="${GAZE_VIDEO_ROOT:-${PROJECT_ROOT}/data/hdepic_vjepa_videos_gaze_sigma${SIGMA}}"
HORIZONS="${HORIZONS:-1}"

if [[ ! -d "${GAZE_VIDEO_ROOT}" ]]; then
  echo "[gaze-val] Missing gaze video root: ${GAZE_VIDEO_ROOT}" >&2
  echo "[gaze-val] Run scripts/prepare_hdepic_gaze_val_tree_cpu.slurm first." >&2
  exit 1
fi

export PROJECT_ROOT SIGMA GAZE_VIDEO_ROOT HORIZONS
export ADAPTED_VIDEO_ROOT="${GAZE_VIDEO_ROOT}"
export CONFIG_DIR="${CONFIG_DIR:-${PROJECT_ROOT}/configs/generated/gaze_sigma${SIGMA}}"

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_val_horizons.slurm"
