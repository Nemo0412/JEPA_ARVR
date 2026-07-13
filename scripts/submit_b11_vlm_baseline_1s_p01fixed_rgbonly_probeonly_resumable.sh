#!/bin/bash
set -euo pipefail

# Resumable variant of submit_b11_vlm_baseline_1s_p01fixed_rgbonly_probeonly.sh
# for unattended relaunch by watch_and_relaunch_vlm_baseline_1s_p01fixed_rgbonly_probeonly.sh.
#
# This baseline replaces the V-JEPA2 encoder/predictor with a frozen CLIP-L/14
# vision encoder; only the attentive probe trains. Per-step GPU compute is tiny
# (a single no_grad CLIP forward over 8 frames), so it cannot hide the
# decord-decode dataloader latency (~7s/batch at bs32/w10) the way encoder-LoRA
# runs do. Job 11151644 (8f, bs8) and job 11174681 (8f4s, bs32) were both killed
# by the cluster's low-GPU-utilization watchdog after ~2h11-2h20m, with 0%
# instantaneous GPU utilization on 95%+ of samples. This mirrors the same
# structural issue and chosen fix already used for the RGB-only V-JEPA2
# baseline (see submit_b11_singleprobe_1s_p01fixed_rgbonly_probeonly_resumable.sh):
# resume from the last checkpoint and resubmit, rather than precompute/cache
# features or fight the watchdog with synthetic GPU load.
#
# Always sets RESUME_CHECKPOINT=1: harmless on a fresh run (no latest.pt yet),
# and required for relaunches to continue from the last completed epoch instead
# of restarting from epoch 0. Defaults TAG/CONFIG_PATH to the in-flight
# 8f4s/bs32/w10 run so a relaunch picks up outputs/.../<tag>/latest.pt.
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

VLM_MODEL_ID="${VLM_MODEL_ID:-openai/clip-vit-large-patch14}"
VLM_MODEL_CLASS="${VLM_MODEL_CLASS:-clip}"
VLM_NUM_FRAMES="${VLM_NUM_FRAMES:-8}"
VLM_TOKEN_MODE="${VLM_TOKEN_MODE:-pooled}"
VLM_IMAGE_SIZE="${VLM_IMAGE_SIZE:-224}"
VLM_TORCH_DTYPE="${VLM_TORCH_DTYPE:-}"
VLM_LOCAL_FILES_ONLY="${VLM_LOCAL_FILES_ONLY:-0}"
VLM_TRUST_REMOTE_CODE="${VLM_TRUST_REMOTE_CODE:-0}"
VLM_GPU_PULSE_ITERS="${VLM_GPU_PULSE_ITERS:-0}"
VLM_GPU_PULSE_SIZE="${VLM_GPU_PULSE_SIZE:-512}"

WORKERS="${EVAL_NUM_WORKERS:-10}"
VAL_WORKERS="${EVAL_VAL_NUM_WORKERS:-${WORKERS}}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
TAG="${LORA_TAG:-hdepic-vlm-clip-1s-p01fixed-rgbonly-probeonly-8f4s-pooled-bs32-w10}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_vlm_baseline_1s_p01fixed_rgbonly_probeonly.yaml}"
SLURM_TIME="${SLURM_TIME:-48:00:00}"
PATIENCE="${EVAL_EARLY_STOPPING_PATIENCE:-3}"

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
export_csv+=",EVAL_MAX_TRAIN_ITERS=0"
export_csv+=",EVAL_NUM_EPOCHS=10"
export_csv+=",EVAL_BEST_METRIC=val-action-acc"
export_csv+=",EVAL_EARLY_STOPPING_PATIENCE=${PATIENCE}"
export_csv+=",RESUME_CHECKPOINT=1"
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

PARTITION="${SLURM_PARTITION:-h100_tandon}"
GRES="${SLURM_GRES:-gpu:h100:1}"
MEM="${SLURM_MEM:-256GB}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-16}"

echo "[submit-vlm-1s-p01fixed-rgbonly-probeonly-resumable] tag=${TAG} resume=1 workers=${WORKERS}/${VAL_WORKERS}"
echo "[submit-vlm-1s-p01fixed-rgbonly-probeonly-resumable] model=${VLM_MODEL_ID} class=${VLM_MODEL_CLASS} frames=${VLM_NUM_FRAMES} token_mode=${VLM_TOKEN_MODE} batch=${BATCH_SIZE}"
echo "[submit-vlm-1s-p01fixed-rgbonly-probeonly-resumable] partition=${PARTITION} gres=${GRES} mem=${MEM} cpus-per-task=${CPUS_PER_TASK}"
sbatch --time="${SLURM_TIME}" --partition="${PARTITION}" --gres="${GRES}" --mem="${MEM}" --cpus-per-task="${CPUS_PER_TASK}" --export="${export_csv}" "${RUN_SCRIPT}"
