#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_pose_smoke
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_pose_smoke_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_pose_smoke_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# B10 pose-only smoke: 300 train iterations, 1 epoch, debug subset optional.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export PROJECT_ROOT
export GAZE_MODE=pose_rnn_fuse
export GAZE_ROOT="${GAZE_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/SLAM-and-Gaze}"
export GAZE_EXTRACT_ROOT="${GAZE_EXTRACT_ROOT:-${PROJECT_ROOT}/outputs/hdepic_gaze_token_gate/extracted}"
export GAZE_SYNC_ROOT="${GAZE_SYNC_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/Videos}"
export POSE_SLAM_ROOT="${POSE_SLAM_ROOT:-${GAZE_ROOT}/P01/SLAM/multi}"
export POSE_MAPPING_JSON="${POSE_MAPPING_JSON:-${POSE_SLAM_ROOT}/vrs_to_multi_slam.json}"
export POSE_FEATURE_SET="${POSE_FEATURE_SET:-pose_6d}"
export GAZE_RNN_USE_VIDEO_TOKEN=0
export EVAL_MAX_LR="${EVAL_MAX_LR:-0.002}"
export EVAL_MAX_TRAIN_ITERS="${EVAL_MAX_TRAIN_ITERS:-300}"
export EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-1}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
export EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-0}"
export GAZE_DECODER_CACHE_READER="${GAZE_DECODER_CACHE_READER:-1}"
export GAZE_DECODER_NUM_THREADS="${GAZE_DECODER_NUM_THREADS:-1}"
export GAZE_HIDDEN_DUMP=0
export LORA_TAG="${LORA_TAG:-hdepic-lora-pose-rnn-smoke-DEBUG_SUBSET}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_rnn_pose_smoke.yaml}"
export DEBUG_SUBSET_PATH="${DEBUG_SUBSET_PATH:-${PROJECT_ROOT}/configs/debug_subset_p01.json}"

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_rnn_pose_train.slurm"
