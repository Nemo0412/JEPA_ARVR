#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_rnn_pose
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=640GB
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_rnn_pose_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_rnn_pose_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# SLAM pose / IMU-derived trajectory fusion at the probe (B10 pose-only branch).
# Encodes clip-relative 6DoF pose features with a BiGRU and concatenates pooled
# pose tokens to the probe cross-attention K/V block.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export PROJECT_ROOT
export GAZE_MODE=pose_rnn_fuse
export GAZE_ROOT="${GAZE_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/SLAM-and-Gaze}"
export GAZE_EXTRACT_ROOT="${GAZE_EXTRACT_ROOT:-${PROJECT_ROOT}/outputs/hdepic_gaze_token_gate/extracted}"
export GAZE_SYNC_ROOT="${GAZE_SYNC_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/Videos}"
export POSE_SLAM_ROOT="${POSE_SLAM_ROOT:-${GAZE_ROOT}/P01/SLAM/multi}"
export POSE_MAPPING_JSON="${POSE_MAPPING_JSON:-${POSE_SLAM_ROOT}/vrs_to_multi_slam.json}"
export POSE_FEATURE_SET="${POSE_FEATURE_SET:-pose_6d}"
export POSE_QUALITY_MIN="${POSE_QUALITY_MIN:-0.0}"
export POSE_HISTORY_SEC="${POSE_HISTORY_SEC:-0}"
export GAZE_RNN_HIDDEN_DIM="${GAZE_RNN_HIDDEN_DIM:-256}"
export GAZE_RNN_NUM_LAYERS="${GAZE_RNN_NUM_LAYERS:-2}"
export GAZE_RNN_NUM_TOKENS="${GAZE_RNN_NUM_TOKENS:-64}"
export GAZE_RNN_DROPOUT="${GAZE_RNN_DROPOUT:-0.1}"
export GAZE_RNN_BIDIR="${GAZE_RNN_BIDIR:-1}"
export GAZE_RNN_TRAJ_LEN="${GAZE_RNN_TRAJ_LEN:-1024}"
export GAZE_RNN_USE_VIDEO_TOKEN=0
export GAZE_LR_MULT="${GAZE_LR_MULT:-2.0}"
export GAZE_HIDDEN_DUMP="${GAZE_HIDDEN_DUMP:-1}"
export EVAL_MAX_LR="${EVAL_MAX_LR:-0.001}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
export LORA_TAG="${LORA_TAG:-hdepic-lora-pose-rnn-1s-w12-bs2-cache}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_rnn_pose.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-12}"
export EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-4}"
export GAZE_DECODER_CACHE_READER="${GAZE_DECODER_CACHE_READER:-1}"
export GAZE_DECODER_NUM_THREADS="${GAZE_DECODER_NUM_THREADS:-2}"

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
