#!/bin/bash
set -euo pipefail

# ViT-L @256 fp32 fulltrain, 1s concat_ar, legacy split, gaze+pose + output
# regularization (confidence_penalty). Advisor feedback 2026-06-17: legacy-split
# results overfit (train action acc reaches 90% by ep10 while val plateaus
# ~37% in the unregularized legacy gaze-nopose reference 10847438) -- try
# regularization. Reuses the same h100 gaze+pose slurm script and LoRA/reg
# hyperparameters as the (wrong-split) p01fixed reg run 10843819, just pointed
# at the legacy CSVs and with early_stopping_patience=0 so train/val curves
# stay directly comparable to the unregularized legacy references (10847438,
# 10910499), which also ran the full 10 epochs.
#
# NOTE: confidence_penalty only takes effect inside the binary_input_adapter
# gaze/pose training loop (binary_input_adapter.py); it is a silent no-op for
# GAZE_MODE=none (RGB-only) runs, which use train_one_epoch_encoder_lora
# instead. Do not reuse LORA_OUTPUT_REG_* env vars on an RGB-only submit
# script expecting them to do anything.
#
# No legacy gaze+pose baseline (reg disabled) exists yet, so this run cannot
# isolate the regularization effect from the pose-vs-no-pose effect on its
# own -- it's an exploratory run on top of the gaze-nopose reference, not a
# controlled A/B. See docs/HD_EPIC_SYNC_NOTES.md.
#
# Code-default note (same regression as the RGB-only baseline): post-8f1adea
# defaults are class_space=phd_reference/temporal_sampling=phd_reference, which
# reject this legacy CSV's action coverage ("PhD-reference action map does not
# cover all validation pairs"). First submission (job 10995823) failed on this
# exact error after ~1 minute. Fixed by explicitly setting
# LORA_CLASS_SPACE=train_only and LORA_TEMPORAL_SAMPLING=legacy below.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_single_probe_encoder_lora_gaze_pose_matrix_h100.slurm"
WORKERS="${EVAL_NUM_WORKERS:-4}"
VAL_WORKERS="${EVAL_VAL_NUM_WORKERS:-${WORKERS}}"
REG_MODE="${LORA_OUTPUT_REG_MODE:-confidence_penalty}"
REG_WEIGHT="${LORA_OUTPUT_REG_WEIGHT:-0.01}"
REG_TARGETS="${LORA_OUTPUT_REG_TARGETS:-verb|noun|action}"
TAG="${LORA_TAG:-hdepic-singleprobe-1s-legacy-gazepose-regcp-w0p01-vitl-fp32-bs8-noac-10ep-w${WORKERS}}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_singleprobe_1s_legacy_gazepose_reg_vitl_fp32_bs8_noac_fulltrain.yaml}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vitl.pt}"
SLURM_TIME="${SLURM_TIME:-72:00:00}"
PATIENCE="${EVAL_EARLY_STOPPING_PATIENCE:-0}"

export_csv="NONE"
export_csv+=",PROJECT_ROOT=${PROJECT_ROOT}"
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
export_csv+=",EVAL_TRAIN_ANTICIPATION_SEC_MIN="
export_csv+=",EVAL_TRAIN_ANTICIPATION_SEC_MAX="
export_csv+=",EVAL_MAX_TRAIN_ITERS=0"
export_csv+=",EVAL_NUM_EPOCHS=10"
export_csv+=",EVAL_VAL_EVERY=1"
export_csv+=",EVAL_BEST_METRIC=val-action-acc"
export_csv+=",EVAL_EARLY_STOPPING_PATIENCE=${PATIENCE}"
export_csv+=",RESUME_CHECKPOINT=0"
export_csv+=",EVAL_SINGLE_PROBE=1"
export_csv+=",LORA_PROBE_TRAIN_MODE=full"
export_csv+=",EVAL_LR=0.0001"
export_csv+=",EVAL_BATCH_SIZE=8"
export_csv+=",EVAL_GRAD_CLIP=1.0"
export_csv+=",EVAL_WARMUP_EPOCHS=2"
export_csv+=",LORA_CLASS_SPACE=train_only"
export_csv+=",LORA_TEMPORAL_SAMPLING=legacy"
export_csv+=",ENCODER_LORA_RANK=8"
export_csv+=",ENCODER_LORA_ALPHA=16.0"
export_csv+=",ENCODER_LORA_LAST_N_BLOCKS=0"
export_csv+=",ENCODER_LORA_LR_MULT=0.5"
export_csv+=",ENCODER_LORA_TARGET_SUFFIXES=attn.qkv|attn.proj"
export_csv+=",ENCODER_LORA_ACTIVATION_CHECKPOINTING=0"
export_csv+=",BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING=0"
export_csv+=",LORA_PRETRAINED_PROBE="
export_csv+=",LORA_OUTPUT_REG_MODE=${REG_MODE}"
export_csv+=",LORA_OUTPUT_REG_WEIGHT=${REG_WEIGHT}"
export_csv+=",LORA_OUTPUT_REG_TARGETS=${REG_TARGETS}"
export_csv+=",LORA_VAL_METRIC_SCOPE=native"
export_csv+=",LORA_VAL_METRIC_AGGREGATION=metric_wise_max"
export_csv+=",EVAL_USE_BFLOAT16=0"
export_csv+=",EVAL_NUM_WORKERS=${WORKERS}"
export_csv+=",EVAL_VAL_NUM_WORKERS=${VAL_WORKERS}"

echo "[submit-vitl-1s-legacy-gazepose-reg-fulltrain] tag=${TAG}"
echo "[submit-vitl-1s-legacy-gazepose-reg-fulltrain] split=legacy via current HD_EPIC_*_vjepa.csv (expect 5744/1640/1045 -- verify in run log); reg=${REG_MODE} weight=${REG_WEIGHT} targets=${REG_TARGETS}; patience=${PATIENCE}"
echo "[submit-vitl-1s-legacy-gazepose-reg-fulltrain] closest unregularized reference: legacy gaze-nopose 10847438 (no pose, no reg, train90/val37 by ep10); no legacy gaze+pose unregularized baseline exists"
sbatch --time="${SLURM_TIME}" --export="${export_csv}" "${RUN_SCRIPT}"
