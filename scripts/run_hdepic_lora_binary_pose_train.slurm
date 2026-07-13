#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_binary_pose_train
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=768GB
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_binary_pose_train_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_binary_pose_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Full P01 train/val: binary gaze map (B5-style) + pose BiGRU (B10) at probe.
# Hyperparams match long-smoke validation (bs2/w12/mem640); no DEBUG_SUBSET.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export PROJECT_ROOT
export GAZE_MODE=binary_input_adapter_pose_rnn_fuse
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
export GAZE_HIDDEN_DUMP=0

export EVAL_MAX_LR="${EVAL_MAX_LR:-0.0015}"
export EVAL_MAX_TRAIN_ITERS="${EVAL_MAX_TRAIN_ITERS:-0}"
export EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-5}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

# H100 node ~1.5TB / 4 GPUs: request 768G for w10 (headroom over 640G OOM; easier to schedule than 1T).
# w12 full still needs ~700GiB+; stay on w10 unless --mem raised further.
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-10}"
export EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-2}"
export GAZE_DECODER_CACHE_READER="${GAZE_DECODER_CACHE_READER:-1}"
export GAZE_DECODER_NUM_THREADS="${GAZE_DECODER_NUM_THREADS:-2}"

export PERF_MONITOR="${PERF_MONITOR:-1}"
export PERF_INTERVAL="${PERF_INTERVAL:-30}"

export DEBUG_SUBSET_PATH=""
export LORA_TAG="${LORA_TAG:-hdepic-lora-binary-gaze-pose-rnn-5ep-bs2-w10-m768-p01}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_binary_gaze_pose_rnn_train.yaml}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
