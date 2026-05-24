#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_tgaze
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_lora_token_gaze_%j.out
#SBATCH --error=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_lora_token_gaze_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"

export PROJECT_ROOT
export GAZE_MODE=token_gate
export GAZE_ROOT="${GAZE_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/SLAM-and-Gaze}"
export GAZE_EXTRACT_ROOT="${GAZE_EXTRACT_ROOT:-${PROJECT_ROOT}/outputs/hdepic_gaze_token_gate/extracted}"
export GAZE_SYNC_ROOT="${GAZE_SYNC_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/HD-EPIC/Videos}"
export GAZE_GATE_GAMMA="${GAZE_GATE_GAMMA:-0.7}"
export GAZE_GATE_SIGMA_PX="${GAZE_GATE_SIGMA_PX:-40}"
export LORA_TAG="${LORA_TAG:-hdepic-lora-token-gaze}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_token_gaze.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-6}"

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
