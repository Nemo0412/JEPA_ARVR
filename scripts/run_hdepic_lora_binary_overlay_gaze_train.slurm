#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_bogaze
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_binary_overlay_gaze_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_binary_overlay_gaze_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
BINARY_RADIUS="${BINARY_RADIUS:-64}"
RADIUS_TAG="${BINARY_RADIUS//./p}"

export PROJECT_ROOT
export GAZE_MODE=overlay_video
export GAZE_VIDEO_ROOT="${GAZE_VIDEO_ROOT:-${PROJECT_ROOT}/data/hdepic_vjepa_videos_gaze_binary_radius${RADIUS_TAG}}"
export LORA_TAG="${LORA_TAG:-hdepic-lora-binary-overlay-gaze-radius${RADIUS_TAG}}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_binary_overlay_gaze_radius${RADIUS_TAG}.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
