#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_rnn_gaze
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=640GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_lora_rnn_gaze_%j.out
#SBATCH --error=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_lora_rnn_gaze_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# RNN gaze fusion at the probe (third gaze variant).
# Encodes the 4s observed gaze trajectory with a BiGRU and concatenates
# pooled gaze tokens to the K/V of the probe's cross-attention block.

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"

export PROJECT_ROOT
export GAZE_MODE=rnn_fuse
export GAZE_ROOT="${GAZE_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/SLAM-and-Gaze}"
export GAZE_EXTRACT_ROOT="${GAZE_EXTRACT_ROOT:-${PROJECT_ROOT}/outputs/hdepic_gaze_token_gate/extracted}"
export GAZE_SYNC_ROOT="${GAZE_SYNC_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/HD-EPIC/Videos}"
export GAZE_RNN_HIDDEN_DIM="${GAZE_RNN_HIDDEN_DIM:-256}"
export GAZE_RNN_NUM_LAYERS="${GAZE_RNN_NUM_LAYERS:-2}"
export GAZE_RNN_NUM_TOKENS="${GAZE_RNN_NUM_TOKENS:-64}"
export GAZE_RNN_DROPOUT="${GAZE_RNN_DROPOUT:-0.1}"
export GAZE_RNN_BIDIR="${GAZE_RNN_BIDIR:-1}"
export GAZE_RNN_TRAJ_LEN="${GAZE_RNN_TRAJ_LEN:-1024}"
export GAZE_RNN_USE_VIDEO_TOKEN="${GAZE_RNN_USE_VIDEO_TOKEN:-1}"
export GAZE_RNN_VIDEO_FUSION="${GAZE_RNN_VIDEO_FUSION:-nearest_concat}"
export GAZE_RNN_VIDEO_PROJ_DIM="${GAZE_RNN_VIDEO_PROJ_DIM:-128}"
export GAZE_RNN_LOCAL_TEMPORAL_RADIUS="${GAZE_RNN_LOCAL_TEMPORAL_RADIUS:-0}"
export GAZE_RNN_LOCAL_SPATIAL_RADIUS="${GAZE_RNN_LOCAL_SPATIAL_RADIUS:-1}"
export GAZE_RNN_RESIDUAL_ALPHA_INIT="${GAZE_RNN_RESIDUAL_ALPHA_INIT:-0.01}"
export GAZE_LR_MULT="${GAZE_LR_MULT:-5.0}"
export GAZE_HIDDEN_DUMP="${GAZE_HIDDEN_DUMP:-1}"
export LORA_TAG="${LORA_TAG:-hdepic-lora-rnn-gaze}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_rnn_gaze.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-6}"

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
