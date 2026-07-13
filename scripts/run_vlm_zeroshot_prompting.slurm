#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__vlm_zeroshot_prompting
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/yh6416/VJEPA2-EXP/logs/vlm_zeroshot_prompting_%j.out
#SBATCH --error=/scratch/yh6416/VJEPA2-EXP/logs/vlm_zeroshot_prompting_%j.err

# Direct zero-shot VLM prompting baseline (no training): runs
# app.hdepic_lora_action_anticipation.zeroshot_vlm_prompting against a chat-capable
# VLM (Llama-3.2-Vision-Instruct or LLaVA-OneVision). Independent of
# run_hdepic_lora_probe.slurm / vlm_video_encoder.py -- does not touch the
# encoder-replacement attentive-probe baselines.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

BACKEND="${BACKEND:?BACKEND is required (llama32vision or llava_onevision)}"
MODEL_ID="${MODEL_ID:-}"
SPLIT_CSV="${SPLIT_CSV:?SPLIT_CSV is required}"
VIDEO_ROOT="${VIDEO_ROOT:?VIDEO_ROOT is required}"
VERB_CLASSES_CSV="${VERB_CLASSES_CSV:?VERB_CLASSES_CSV is required}"
NOUN_CLASSES_CSV="${NOUN_CLASSES_CSV:?NOUN_CLASSES_CSV is required}"
OUTPUT_CSV="${OUTPUT_CSV:?OUTPUT_CSV is required}"
NUM_FRAMES="${NUM_FRAMES:-32}"
TARGET_FPS="${TARGET_FPS:-8.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LOG_EVERY="${LOG_EVERY:-20}"
LOCAL_FILES_ONLY_FLAG=""
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  LOCAL_FILES_ONLY_FLAG="--local-files-only"
fi
FEW_SHOT_K="${FEW_SHOT_K:-0}"
FEW_SHOT_CSV="${FEW_SHOT_CSV:-}"
FEW_SHOT_SEED="${FEW_SHOT_SEED:-0}"
FEW_SHOT_NUM_FRAMES="${FEW_SHOT_NUM_FRAMES:-8}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"

export HF_HOME="${HF_HOME:-/scratch/yh6416/.huggingface}"
export PROJECT_ROOT ENV_NAME BACKEND MODEL_ID SPLIT_CSV VIDEO_ROOT VERB_CLASSES_CSV NOUN_CLASSES_CSV OUTPUT_CSV
export NUM_FRAMES TARGET_FPS MAX_NEW_TOKENS MAX_SAMPLES TORCH_DTYPE LOG_EVERY LOCAL_FILES_ONLY_FLAG
export FEW_SHOT_K FEW_SHOT_CSV FEW_SHOT_SEED FEW_SHOT_NUM_FRAMES LORA_ADAPTER_PATH

echo "[zeroshot] backend=${BACKEND} model_id=${MODEL_ID:-<default>} split_csv=${SPLIT_CSV}"
echo "[zeroshot] num_frames=${NUM_FRAMES} target_fps=${TARGET_FPS} max_new_tokens=${MAX_NEW_TOKENS} max_samples=${MAX_SAMPLES} dtype=${TORCH_DTYPE}"
echo "[zeroshot] few_shot_k=${FEW_SHOT_K} few_shot_csv=${FEW_SHOT_CSV} few_shot_seed=${FEW_SHOT_SEED}"
echo "[zeroshot] output_csv=${OUTPUT_CSV}"

payload=""
read -r -d '' payload <<'BASH' || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

mkdir -p "$(dirname "${OUTPUT_CSV}")"

few_shot_args=()
if [[ "${FEW_SHOT_K}" != "0" ]]; then
  few_shot_args=(--few-shot-k "${FEW_SHOT_K}" --few-shot-csv "${FEW_SHOT_CSV}" --few-shot-seed "${FEW_SHOT_SEED}" --few-shot-num-frames "${FEW_SHOT_NUM_FRAMES}")
fi
lora_args=()
if [[ -n "${LORA_ADAPTER_PATH}" ]]; then
  lora_args=(--lora-adapter-path "${LORA_ADAPTER_PATH}")
fi

python -m app.hdepic_lora_action_anticipation.zeroshot_vlm_prompting \
  --backend "${BACKEND}" \
  ${MODEL_ID:+--model-id "${MODEL_ID}"} \
  --split-csv "${SPLIT_CSV}" \
  --video-root "${VIDEO_ROOT}" \
  --verb-classes-csv "${VERB_CLASSES_CSV}" \
  --noun-classes-csv "${NOUN_CLASSES_CSV}" \
  --output-csv "${OUTPUT_CSV}" \
  --num-frames "${NUM_FRAMES}" \
  --target-fps "${TARGET_FPS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-samples "${MAX_SAMPLES}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --log-every "${LOG_EVERY}" \
  ${LOCAL_FILES_ONLY_FLAG} \
  "${few_shot_args[@]}" \
  "${lora_args[@]}"
BASH

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
