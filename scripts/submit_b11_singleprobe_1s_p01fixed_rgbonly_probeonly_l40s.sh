#!/bin/bash
set -euo pipefail

# L40S variant of the p01_fixed protocol RGB-only probe-only baseline
# (hdepic_singleprobe_1s_p01fixed_rgbonly_probeonly_vitl_fp32_bs8_noac_fulltrain).
# Runs alongside the H100 run of the same config (job 10918458) as a separate,
# non-urgent copy intended as an inference baseline checkpoint for the LTM branch --
# does not need to finish cleanly or stay alive; if the cluster's low-GPU-utilization
# watchdog kills it (same risk as the H100 run: encoder frozen, GPU compute per step
# too small to hide dataloader latency), that's fine, just resubmit this script with
# RESUME_CHECKPOINT=1 (already default here) to continue from the last checkpoint.
#
# Uses a distinct LORA_TAG (suffix -l40s) so its output dir/checkpoints don't collide
# with the H100 comparison run's tag.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
ADAPTED_ANNOTATIONS_DIR="${ADAPTED_ANNOTATIONS_DIR:-${PROJECT_ROOT}/data/hdepic_vjepa_annotations/p01_fixed}"
HDEPIC_ANN_ROOT="${HDEPIC_ANN_ROOT:-${PROJECT_ROOT}/data/hd-epic-annotations/narrations-and-action-segments}"
WORKERS="${EVAL_NUM_WORKERS:-10}"
VAL_WORKERS="${EVAL_VAL_NUM_WORKERS:-${WORKERS}}"
TAG="${LORA_TAG:-hdepic-singleprobe-1s-p01fixed-rgbonly-probeonly-vitl-fp32-bs8-noac-10ep-l40s}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_singleprobe_1s_p01fixed_rgbonly_probeonly_l40s_vitl_fp32_bs8_noac_fulltrain.yaml}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vitl.pt}"
SLURM_TIME="${SLURM_TIME:-48:00:00}"
PATIENCE="${EVAL_EARLY_STOPPING_PATIENCE:-3}"

export_csv="ALL"
export_csv+=",PROJECT_ROOT=${PROJECT_ROOT}"
export_csv+=",ADAPTED_ANNOTATIONS_DIR=${ADAPTED_ANNOTATIONS_DIR}"
export_csv+=",HDEPIC_REFERENCE_ANNOTATIONS_PKL=${HDEPIC_ANN_ROOT}/HD_EPIC_Narrations.pkl"
export_csv+=",HDEPIC_REFERENCE_VERB_CLASSES_CSV=${HDEPIC_ANN_ROOT}/HD_EPIC_verb_classes.csv"
export_csv+=",HDEPIC_REFERENCE_NOUN_CLASSES_CSV=${HDEPIC_ANN_ROOT}/HD_EPIC_noun_classes.csv"
export_csv+=",LORA_TAG=${TAG}"
export_csv+=",CONFIG_PATH=${CONFIG_PATH}"
export_csv+=",CHECKPOINT=${CHECKPOINT}"
export_csv+=",BACKBONE=vitl"
export_csv+=",EVAL_RESOLUTION=256"
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
export_csv+=",EVAL_BATCH_SIZE=8"
export_csv+=",EVAL_GRAD_CLIP=1.0"
export_csv+=",EVAL_WARMUP_EPOCHS=2"
export_csv+=",GAZE_MODE=none"
export_csv+=",ENCODER_LORA_ENABLED=0"
export_csv+=",BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING=0"
export_csv+=",LORA_PRETRAINED_PROBE="
export_csv+=",LORA_VAL_METRIC_SCOPE=native"
export_csv+=",EVAL_USE_BFLOAT16=0"
export_csv+=",EVAL_USE_FOCAL_LOSS=0"
export_csv+=",EVAL_NUM_WORKERS=${WORKERS}"
export_csv+=",EVAL_VAL_NUM_WORKERS=${VAL_WORKERS}"
export_csv+=",PERF_MONITOR=1"

PARTITION="${SLURM_PARTITION:-l40s_public}"
GRES="${SLURM_GRES:-gpu:l40s:1}"
MEM="${SLURM_MEM:-240GB}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-12}"

echo "[submit-vitl-1s-p01fixed-rgbonly-probeonly-l40s] tag=${TAG} (separate from H100 comparison run; non-urgent LTM inference-baseline copy)"
echo "[submit-vitl-1s-p01fixed-rgbonly-probeonly-l40s] partition=${PARTITION} gres=${GRES} mem=${MEM} cpus-per-task=${CPUS_PER_TASK} workers=${WORKERS}/${VAL_WORKERS}"
sbatch --time="${SLURM_TIME}" --partition="${PARTITION}" --gres="${GRES}" --mem="${MEM}" --cpus-per-task="${CPUS_PER_TASK}" --export="${export_csv}" "${RUN_SCRIPT}"
