#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/ENCLO-LATENCY-VJEPA2}"
MAIN_ROOT="${MAIN_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"

PATH_KIND="${1:-baseline}"
BATCH_SIZE=4
ENCODER_AC=1
ADAPTER_AC=1
DIRECT_BWD=0
BACKBONE="${BACKBONE:-vitg}"
CHECKPOINT="${CHECKPOINT:-${MAIN_ROOT}/checkpoints/vitg-384.pt}"
EVAL_RESOLUTION="${EVAL_RESOLUTION:-384}"
# One yaml per path kind — avoids races when multiple latency jobs start together.
CONFIG_PATH="${PROJECT_ROOT}/configs/generated/latency_${PATH_KIND}.yaml"

case "${PATH_KIND}" in
  baseline)
    TAG="hdepic-enclora-fp32-latency-baseline-i30-v3"
    GAZE_MODE="none"
    EXTRA_ENV="EVAL_SINGLE_PROBE=1,LORA_PROBE_TRAIN_MODE=full,LORA_PRETRAINED_PROBE="
    ;;
  singleprobe)
    TAG="hdepic-enclora-fp32-latency-singleprobe-i30-v3"
    GAZE_MODE="binary_input_adapter_gaze_pose_matrix"
    EXTRA_ENV="EVAL_SINGLE_PROBE=1,LORA_PROBE_TRAIN_MODE=full,LORA_PRETRAINED_PROBE=,BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING=${ADAPTER_AC},BINARY_INPUT_ADAPTER_DIRECT_BACKWARD=${DIRECT_BWD}"
    ;;
  singleprobe-bs4-noac)
    TAG="hdepic-enclora-fp32-latency-singleprobe-bs4-noac-vitl-i30"
    GAZE_MODE="binary_input_adapter_gaze_pose_matrix"
    BACKBONE="vitl"
    CHECKPOINT="${MAIN_ROOT}/checkpoints/vitl.pt"
    EVAL_RESOLUTION=256
    ENCODER_AC=0
    ADAPTER_AC=0
    EXTRA_ENV="EVAL_SINGLE_PROBE=1,LORA_PROBE_TRAIN_MODE=full,LORA_PRETRAINED_PROBE=,BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING=${ADAPTER_AC},BINARY_INPUT_ADAPTER_DIRECT_BACKWARD=${DIRECT_BWD},BACKBONE=${BACKBONE},EVAL_RESOLUTION=${EVAL_RESOLUTION}"
    ;;
  singleprobe-bs8-noac)
    TAG="hdepic-enclora-fp32-latency-singleprobe-bs8-noac-vitl-i30"
    GAZE_MODE="binary_input_adapter_gaze_pose_matrix"
    BACKBONE="vitl"
    CHECKPOINT="${MAIN_ROOT}/checkpoints/vitl.pt"
    EVAL_RESOLUTION=256
    BATCH_SIZE=8
    ENCODER_AC=0
    ADAPTER_AC=0
    EXTRA_ENV="EVAL_SINGLE_PROBE=1,LORA_PROBE_TRAIN_MODE=full,LORA_PRETRAINED_PROBE=,BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING=${ADAPTER_AC},BINARY_INPUT_ADAPTER_DIRECT_BACKWARD=${DIRECT_BWD},BACKBONE=${BACKBONE},EVAL_RESOLUTION=${EVAL_RESOLUTION}"
    ;;
  singleprobe-direct-bs4-noac)
    TAG="hdepic-enclora-fp32-latency-singleprobe-direct-bs4-noac-vitl-i30"
    GAZE_MODE="binary_input_adapter_gaze_pose_matrix"
    BACKBONE="vitl"
    CHECKPOINT="${MAIN_ROOT}/checkpoints/vitl.pt"
    EVAL_RESOLUTION=256
    ENCODER_AC=0
    ADAPTER_AC=0
    DIRECT_BWD=1
    EXTRA_ENV="EVAL_SINGLE_PROBE=1,LORA_PROBE_TRAIN_MODE=full,LORA_PRETRAINED_PROBE=,BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING=${ADAPTER_AC},BINARY_INPUT_ADAPTER_DIRECT_BACKWARD=${DIRECT_BWD},BACKBONE=${BACKBONE},EVAL_RESOLUTION=${EVAL_RESOLUTION}"
    ;;
  *)
    echo "Usage: $0 [baseline|singleprobe|singleprobe-bs4-noac|singleprobe-bs8-noac|singleprobe-direct-bs4-noac]" >&2
    exit 1
    ;;
esac

REPORT_PATH="${PROJECT_ROOT}/logs/latency_${TAG}.json"
export_csv="ALL,PROJECT_ROOT=${PROJECT_ROOT},MAIN_ROOT=${MAIN_ROOT},OVERLAY=${MAIN_ROOT}/overlay-15GB-500K.ext3,ADAPTED_ANNOTATIONS_DIR=${MAIN_ROOT}/data/hdepic_vjepa_annotations,ADAPTED_VIDEO_ROOT=${MAIN_ROOT}/data/hdepic_vjepa_videos,CHECKPOINT=${CHECKPOINT},GAZE_ROOT=${MAIN_ROOT}/data/raw/HD-EPIC/SLAM-and-Gaze,GAZE_SYNC_ROOT=${MAIN_ROOT}/data/raw/HD-EPIC/Videos,GAZE_EXTRACT_ROOT=${MAIN_ROOT}/outputs/hdepic_gaze_token_gate/extracted,GAZE_MODE=${GAZE_MODE},GAZE_DECODER_NUM_THREADS=2,ENCODER_LORA_ENABLED=1,ENCODER_LORA_RANK=8,ENCODER_LORA_ALPHA=16.0,ENCODER_LORA_LAST_N_BLOCKS=0,ENCODER_LORA_LR_MULT=0.5,ENCODER_LORA_TARGET_SUFFIXES=attn.qkv|attn.proj,ENCODER_LORA_ACTIVATION_CHECKPOINTING=${ENCODER_AC},EVAL_USE_BFLOAT16=0,EVAL_BATCH_SIZE=${BATCH_SIZE},EVAL_NUM_WORKERS=2,EVAL_MAX_TRAIN_ITERS=30,EVAL_NUM_EPOCHS=1,EVAL_GRAD_CLIP=1.0,EVAL_WARMUP_EPOCHS=2,EVAL_LATENCY_BREAKDOWN=1,EVAL_LATENCY_LOG_INTERVAL=10,EVAL_LATENCY_REPORT=${REPORT_PATH},LORA_TAG=${TAG},CONFIG_PATH=${CONFIG_PATH},RESUME_CHECKPOINT=0,OUTPUT_DIR=${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation,BACKBONE=${BACKBONE},EVAL_RESOLUTION=${EVAL_RESOLUTION},${EXTRA_ENV}"

echo "[submit-latency] path=${PATH_KIND} tag=${TAG} config=${CONFIG_PATH} backbone=${BACKBONE} res=${EVAL_RESOLUTION} fp32 bs=${BATCH_SIZE} enc_ac=${ENCODER_AC} adapter_ac=${ADAPTER_AC} direct_bwd=${DIRECT_BWD}"
sbatch \
  --partition=h100_tandon \
  --account=your_slurm_account \
  --gres=gpu:h100:1 \
  --cpus-per-task=12 \
  --mem=768GB \
  --time=04:00:00 \
  --job-name="VJEPA2-EXP__latency_${PATH_KIND}" \
  --output="${PROJECT_ROOT}/logs/hdepic_enclora_fp32_latency_${PATH_KIND}_%j.out" \
  --error="${PROJECT_ROOT}/logs/hdepic_enclora_fp32_latency_${PATH_KIND}_%j.err" \
  --export="${export_csv}" \
  "${RUN_SCRIPT}"
echo "[submit-latency] submitted ${PATH_KIND}"
