#!/bin/bash
set -euo pipefail

# Submit wrapper for the zero-shot VLM prompting baseline (no training).
#
# Usage:
#   BACKEND=llama32vision bash scripts/submit_vlm_zeroshot_prompting.sh
#   BACKEND=llava_onevision bash scripts/submit_vlm_zeroshot_prompting.sh
#
# Smoke test (few samples, short time limit):
#   BACKEND=llama32vision MAX_SAMPLES=5 SLURM_TIME=00:30:00 \
#     bash scripts/submit_vlm_zeroshot_prompting.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_vlm_zeroshot_prompting.slurm"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"

BACKEND="${BACKEND:?Set BACKEND=llama32vision or BACKEND=llava_onevision}"
MODEL_ID="${MODEL_ID:-}"
SPLIT="${SPLIT:-test}"
SPLIT_CSV="${SPLIT_CSV:-${SHARED_PROJECT_ROOT}/data/hdepic_vjepa_annotations/p01_fixed/HD_EPIC_${SPLIT}_vjepa.csv}"
VIDEO_ROOT="${VIDEO_ROOT:-${SHARED_PROJECT_ROOT}/data/hdepic_vjepa_videos}"
HDEPIC_ANN_ROOT="${HDEPIC_ANN_ROOT:-${SHARED_PROJECT_ROOT}/data/hd-epic-annotations/narrations-and-action-segments}"
VERB_CLASSES_CSV="${VERB_CLASSES_CSV:-${HDEPIC_ANN_ROOT}/HD_EPIC_verb_classes.csv}"
NOUN_CLASSES_CSV="${NOUN_CLASSES_CSV:-${HDEPIC_ANN_ROOT}/HD_EPIC_noun_classes.csv}"
NUM_FRAMES="${NUM_FRAMES:-32}"
TARGET_FPS="${TARGET_FPS:-8.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LOG_EVERY="${LOG_EVERY:-20}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
FEW_SHOT_K="${FEW_SHOT_K:-0}"
FEW_SHOT_CSV="${FEW_SHOT_CSV:-${SHARED_PROJECT_ROOT}/data/hdepic_vjepa_annotations/p01_fixed/HD_EPIC_train_vjepa.csv}"
FEW_SHOT_SEED="${FEW_SHOT_SEED:-0}"
FEW_SHOT_NUM_FRAMES="${FEW_SHOT_NUM_FRAMES:-8}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"
# Tag output paths by frame count, few-shot-k, and lora tag so different
# configurations don't silently overwrite each other's predictions/summary on disk.
LORA_TAG="${LORA_TAG:-}"
OUTPUT_CSV="${OUTPUT_CSV:-${SHARED_PROJECT_ROOT}/outputs/vlm_zeroshot_prompting/${BACKEND}_${SPLIT}_${NUM_FRAMES}f_fewshot${FEW_SHOT_K}${LORA_TAG:+_${LORA_TAG}}_predictions.csv}"

export_csv="ALL"
export_csv+=",PROJECT_ROOT=${PROJECT_ROOT}"
export_csv+=",SHARED_PROJECT_ROOT=${SHARED_PROJECT_ROOT}"
export_csv+=",OVERLAY=${OVERLAY}"
export_csv+=",BACKEND=${BACKEND}"
export_csv+=",MODEL_ID=${MODEL_ID}"
export_csv+=",SPLIT_CSV=${SPLIT_CSV}"
export_csv+=",VIDEO_ROOT=${VIDEO_ROOT}"
export_csv+=",VERB_CLASSES_CSV=${VERB_CLASSES_CSV}"
export_csv+=",NOUN_CLASSES_CSV=${NOUN_CLASSES_CSV}"
export_csv+=",OUTPUT_CSV=${OUTPUT_CSV}"
export_csv+=",NUM_FRAMES=${NUM_FRAMES}"
export_csv+=",TARGET_FPS=${TARGET_FPS}"
export_csv+=",MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
export_csv+=",MAX_SAMPLES=${MAX_SAMPLES}"
export_csv+=",TORCH_DTYPE=${TORCH_DTYPE}"
export_csv+=",LOG_EVERY=${LOG_EVERY}"
export_csv+=",LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY}"
export_csv+=",FEW_SHOT_K=${FEW_SHOT_K}"
export_csv+=",FEW_SHOT_CSV=${FEW_SHOT_CSV}"
export_csv+=",FEW_SHOT_SEED=${FEW_SHOT_SEED}"
export_csv+=",FEW_SHOT_NUM_FRAMES=${FEW_SHOT_NUM_FRAMES}"
export_csv+=",LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH}"
export_csv+=",HF_HOME=${HF_HOME:-/scratch/yh6416/.huggingface}"

PARTITION="${SLURM_PARTITION:-h100_tandon}"
GRES="${SLURM_GRES:-gpu:h100:1}"
MEM="${SLURM_MEM:-128GB}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

echo "[submit-vlm-zeroshot] backend=${BACKEND} split=${SPLIT} max_samples=${MAX_SAMPLES} few_shot_k=${FEW_SHOT_K}"
echo "[submit-vlm-zeroshot] output_csv=${OUTPUT_CSV}"
echo "[submit-vlm-zeroshot] partition=${PARTITION} gres=${GRES} mem=${MEM} cpus-per-task=${CPUS_PER_TASK} time=${SLURM_TIME}"
sbatch --time="${SLURM_TIME}" --partition="${PARTITION}" --gres="${GRES}" --mem="${MEM}" --cpus-per-task="${CPUS_PER_TASK}" --export="${export_csv}" "${RUN_SCRIPT}"
