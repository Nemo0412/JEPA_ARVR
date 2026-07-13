#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_mlp_gaze
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_mlp_gaze_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_mlp_gaze_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Per-frame MLP ablation of the RNN gaze fusion variant. Same trajectory
# loader, same K-token cross-attention injection, same modality embedding,
# but no temporal modeling -- isolates whether the trajectory encoding (vs.
# merely having gaze as a separate modality) contributes the gain.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export PROJECT_ROOT
export GAZE_MODE=mlp_fuse
export GAZE_ROOT="${GAZE_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/SLAM-and-Gaze}"
export GAZE_EXTRACT_ROOT="${GAZE_EXTRACT_ROOT:-${PROJECT_ROOT}/outputs/hdepic_gaze_token_gate/extracted}"
export GAZE_SYNC_ROOT="${GAZE_SYNC_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/Videos}"
export GAZE_RNN_HIDDEN_DIM="${GAZE_RNN_HIDDEN_DIM:-256}"
export GAZE_RNN_NUM_LAYERS="${GAZE_RNN_NUM_LAYERS:-2}"
export GAZE_RNN_NUM_TOKENS="${GAZE_RNN_NUM_TOKENS:-64}"
export GAZE_RNN_DROPOUT="${GAZE_RNN_DROPOUT:-0.1}"
export GAZE_RNN_BIDIR="${GAZE_RNN_BIDIR:-1}"
export GAZE_RNN_TRAJ_LEN="${GAZE_RNN_TRAJ_LEN:-1024}"
export GAZE_LR_MULT="${GAZE_LR_MULT:-5.0}"
export GAZE_HIDDEN_DUMP="${GAZE_HIDDEN_DUMP:-0}"
export LORA_TAG="${LORA_TAG:-hdepic-lora-mlp-gaze}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_mlp_gaze.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-6}"

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
