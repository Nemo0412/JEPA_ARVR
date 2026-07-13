#!/bin/bash
set -euo pipefail

# Llama 3.2 Vision (Mllama) variant of submit_b11_vlm_baseline_1s_p01fixed_rgbonly_probeonly.sh:
# replaces CLIP-L/14 with the frozen meta-llama/Llama-3.2-11B-Vision-Instruct vision
# tower (model_class=mllama in vlm_video_encoder.py), gated HF repo -- requires a
# valid token at $HF_HOME/token (this env's HF_HOME is /scratch/yh6416/.huggingface,
# not the default ~/.cache/huggingface).
#
# Mllama tiles images by aspect ratio and needs aspect_ratio_ids/aspect_ratio_mask;
# vlm_video_encoder.py forces every frame to a single 1x1 (untiled) tile at the
# model's native image_size=560 instead of reproducing Meta's multi-tile splitting.
# VLM_IMAGE_SIZE is included here for documentation but the wrapper overrides it to
# the checkpoint's vision_config.image_size regardless, since the tile position
# embeddings are precomputed for that fixed resolution.
#
# Loads only the vision tower (MllamaVisionModel, ~per vision_config: 32 layers +
# 8 global layers, hidden_size=1280) rather than the full ~10B-parameter
# conditional-generation model, but the safetensors shards on disk still bundle
# vision+text weights together, so the first run downloads the full ~22GB
# checkpoint regardless of how much of it gets loaded into the model object.
# Run a short smoke test first (e.g. EVAL_MAX_TRAIN_ITERS=2 EVAL_NUM_EPOCHS=1
# SLURM_TIME=00:30:00) before committing to the full 10-epoch run, both to warm
# the HF cache and to confirm the aspect-ratio-tiling shapes are right end-to-end.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
ADAPTED_VIDEO_ROOT="${ADAPTED_VIDEO_ROOT:-${SHARED_PROJECT_ROOT}/data/hdepic_vjepa_videos}"
ADAPTED_ANNOTATIONS_DIR="${ADAPTED_ANNOTATIONS_DIR:-${SHARED_PROJECT_ROOT}/data/hdepic_vjepa_annotations/p01_fixed}"
HDEPIC_ANN_ROOT="${HDEPIC_ANN_ROOT:-${SHARED_PROJECT_ROOT}/data/hd-epic-annotations/narrations-and-action-segments}"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"
OUTPUT_DIR="${OUTPUT_DIR:-${SHARED_PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"

VLM_MODEL_ID="${VLM_MODEL_ID:-meta-llama/Llama-3.2-11B-Vision-Instruct}"
VLM_MODEL_CLASS="${VLM_MODEL_CLASS:-mllama}"
VLM_NUM_FRAMES="${VLM_NUM_FRAMES:-8}"
VLM_TOKEN_MODE="${VLM_TOKEN_MODE:-pooled}"
VLM_IMAGE_SIZE="${VLM_IMAGE_SIZE:-560}"
VLM_TORCH_DTYPE="${VLM_TORCH_DTYPE:-bfloat16}"
VLM_LOCAL_FILES_ONLY="${VLM_LOCAL_FILES_ONLY:-0}"
VLM_TRUST_REMOTE_CODE="${VLM_TRUST_REMOTE_CODE:-0}"
VLM_GPU_PULSE_ITERS="${VLM_GPU_PULSE_ITERS:-0}"
VLM_GPU_PULSE_SIZE="${VLM_GPU_PULSE_SIZE:-512}"

WORKERS="${EVAL_NUM_WORKERS:-10}"
VAL_WORKERS="${EVAL_VAL_NUM_WORKERS:-${WORKERS}}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
TAG="${LORA_TAG:-hdepic-vlm-llama32vision11b-1s-p01fixed-rgbonly-probeonly-8f4s-pooled-bs${BATCH_SIZE}-w${WORKERS}}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_vlm_baseline_1s_p01fixed_rgbonly_probeonly_llama32vision.yaml}"
SLURM_TIME="${SLURM_TIME:-48:00:00}"
PATIENCE="${EVAL_EARLY_STOPPING_PATIENCE:-3}"
MAX_TRAIN_ITERS="${EVAL_MAX_TRAIN_ITERS:-0}"
NUM_EPOCHS="${EVAL_NUM_EPOCHS:-10}"
RESUME="${RESUME_CHECKPOINT:-0}"

export_csv="ALL"
export_csv+=",PROJECT_ROOT=${PROJECT_ROOT}"
export_csv+=",ADAPTED_VIDEO_ROOT=${ADAPTED_VIDEO_ROOT}"
export_csv+=",ADAPTED_ANNOTATIONS_DIR=${ADAPTED_ANNOTATIONS_DIR}"
export_csv+=",OVERLAY=${OVERLAY}"
export_csv+=",OUTPUT_DIR=${OUTPUT_DIR}"
export_csv+=",HDEPIC_REFERENCE_ANNOTATIONS_PKL=${HDEPIC_ANN_ROOT}/HD_EPIC_Narrations.pkl"
export_csv+=",HDEPIC_REFERENCE_VERB_CLASSES_CSV=${HDEPIC_ANN_ROOT}/HD_EPIC_verb_classes.csv"
export_csv+=",HDEPIC_REFERENCE_NOUN_CLASSES_CSV=${HDEPIC_ANN_ROOT}/HD_EPIC_noun_classes.csv"
export_csv+=",LORA_TAG=${TAG}"
export_csv+=",CONFIG_PATH=${CONFIG_PATH}"
export_csv+=",CHECKPOINT=${VLM_MODEL_ID}"
export_csv+=",MODEL_FAMILY=vlm"
export_csv+=",VLM_MODEL_ID=${VLM_MODEL_ID}"
export_csv+=",VLM_MODEL_CLASS=${VLM_MODEL_CLASS}"
export_csv+=",VLM_NUM_FRAMES=${VLM_NUM_FRAMES}"
export_csv+=",VLM_TOKEN_MODE=${VLM_TOKEN_MODE}"
export_csv+=",VLM_IMAGE_SIZE=${VLM_IMAGE_SIZE}"
export_csv+=",VLM_TORCH_DTYPE=${VLM_TORCH_DTYPE}"
export_csv+=",VLM_LOCAL_FILES_ONLY=${VLM_LOCAL_FILES_ONLY}"
export_csv+=",VLM_TRUST_REMOTE_CODE=${VLM_TRUST_REMOTE_CODE}"
export_csv+=",VLM_GPU_PULSE_ITERS=${VLM_GPU_PULSE_ITERS}"
export_csv+=",VLM_GPU_PULSE_SIZE=${VLM_GPU_PULSE_SIZE}"
export_csv+=",EVAL_RESOLUTION=${VLM_IMAGE_SIZE}"
export_csv+=",DEBUG_SUBSET_PATH="
export_csv+=",HDEPIC_DIRECT_ROPE_10S=0"
export_csv+=",HDEPIC_AR10S_SLIDING_GAZE_POSE=0"
export_csv+=",AR10S_SLIDING_PROBE=0"
export_csv+=",EVAL_ANTICIPATION_SEC="
export_csv+=",EVAL_MAX_TRAIN_ITERS=${MAX_TRAIN_ITERS}"
export_csv+=",EVAL_NUM_EPOCHS=${NUM_EPOCHS}"
export_csv+=",EVAL_BEST_METRIC=val-action-acc"
export_csv+=",EVAL_EARLY_STOPPING_PATIENCE=${PATIENCE}"
export_csv+=",RESUME_CHECKPOINT=${RESUME}"
export_csv+=",EVAL_SINGLE_PROBE=1"
export_csv+=",LORA_PROBE_TRAIN_MODE=full"
export_csv+=",EVAL_LR=0.0001"
export_csv+=",EVAL_BATCH_SIZE=${BATCH_SIZE}"
export_csv+=",EVAL_GRAD_CLIP=1.0"
export_csv+=",EVAL_WARMUP_EPOCHS=2"
export_csv+=",GAZE_MODE=none"
export_csv+=",ENCODER_LORA_ENABLED=0"
export_csv+=",PREDICTOR_LORA_ENABLED=0"
export_csv+=",BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING=0"
export_csv+=",LORA_PRETRAINED_PROBE="
export_csv+=",LORA_VAL_METRIC_SCOPE=native"
export_csv+=",EVAL_USE_BFLOAT16=0"
export_csv+=",EVAL_USE_FOCAL_LOSS=0"
export_csv+=",EVAL_NUM_WORKERS=${WORKERS}"
export_csv+=",EVAL_VAL_NUM_WORKERS=${VAL_WORKERS}"
export_csv+=",PERF_MONITOR=1"
export_csv+=",HF_HOME=${HF_HOME:-/scratch/yh6416/.huggingface}"

PARTITION="${SLURM_PARTITION:-h100_tandon}"
GRES="${SLURM_GRES:-gpu:h100:1}"
MEM="${SLURM_MEM:-256GB}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-16}"

echo "[submit-vlm-1s-p01fixed-rgbonly-probeonly-llama32vision] tag=${TAG}"
echo "[submit-vlm-1s-p01fixed-rgbonly-probeonly-llama32vision] code_root=${PROJECT_ROOT} shared_root=${SHARED_PROJECT_ROOT}"
echo "[submit-vlm-1s-p01fixed-rgbonly-probeonly-llama32vision] model=${VLM_MODEL_ID} class=${VLM_MODEL_CLASS} frames=${VLM_NUM_FRAMES} token_mode=${VLM_TOKEN_MODE} image_size=${VLM_IMAGE_SIZE} dtype=${VLM_TORCH_DTYPE} batch=${BATCH_SIZE}"
echo "[submit-vlm-1s-p01fixed-rgbonly-probeonly-llama32vision] max_train_iters=${MAX_TRAIN_ITERS} num_epochs=${NUM_EPOCHS} resume=${RESUME}"
echo "[submit-vlm-1s-p01fixed-rgbonly-probeonly-llama32vision] split=p01_fixed; MODEL_FAMILY=vlm; GAZE_MODE=none; encoder/predictor LoRA disabled"
echo "[submit-vlm-1s-p01fixed-rgbonly-probeonly-llama32vision] partition=${PARTITION} gres=${GRES} mem=${MEM} cpus-per-task=${CPUS_PER_TASK} workers=${WORKERS}/${VAL_WORKERS}"
sbatch --time="${SLURM_TIME}" --partition="${PARTITION}" --gres="${GRES}" --mem="${MEM}" --cpus-per-task="${CPUS_PER_TASK}" --export="${export_csv}" "${RUN_SCRIPT}"
