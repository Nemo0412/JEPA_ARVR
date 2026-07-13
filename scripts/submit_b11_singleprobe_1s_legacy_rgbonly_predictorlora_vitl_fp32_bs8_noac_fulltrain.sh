#!/bin/bash
set -euo pipefail

# ViT-L @256 fp32 fulltrain, 1s concat_ar, legacy split, RGB-only, PREDICTOR
# LoRA instead of encoder LoRA. Advisor feedback 2026-06-17: try training the
# predictor with the same LoRA strategy used on the encoder. This is the
# validation run for that new code path (app/hdepic_lora_action_anticipation/
# predictor_lora.py): encoder stays fully frozen (ENCODER_LORA_ENABLED=0),
# only the predictor's attn.qkv/attn.proj get LoRA, same
# rank/alpha/dropout/lr_mult/weight_decay as the existing encoder-LoRA
# baselines (rank=8 alpha=16 dropout=0.05 lr_mult=0.5).
#
# Direct comparison target: legacy RGB-only encoder-LoRA baseline (job
# 10910499, tag hdepic-singleprobe-1s-legacy-rgbonly-...-w10). Same split,
# same data/optimization hyperparameters, same resource shape (w10/256GB/16cpu,
# sized from that job's relaunch history) -- only the LoRA target (encoder vs
# predictor) differs, so any metric delta should be attributable to that.
#
# Code-default note: same legacy-split bypass as the RGB-only baseline --
# post-8f1adea defaults (class_space=phd_reference, temporal_sampling=
# phd_reference) reject this legacy CSV's action coverage, so this run also
# sets LORA_CLASS_SPACE=train_only and LORA_TEMPORAL_SAMPLING=legacy.
#
# NOTE: the predictor does not get a `latest.pt`-style sidecar checkpoint on
# this baseline (no-gaze) path -- same known gap as encoder LoRA on the
# RGB-only path (save_*_lora_checkpoint is only wired into the gaze/binary
# adapter training loop). Fine for a single continuous training+test run;
# would need code if a later experiment needs job-level resume of predictor
# LoRA weights specifically.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
WORKERS="${EVAL_NUM_WORKERS:-10}"
VAL_WORKERS="${EVAL_VAL_NUM_WORKERS:-${WORKERS}}"
TAG="${LORA_TAG:-hdepic-singleprobe-1s-legacy-rgbonly-predictorlora-vitl-fp32-bs8-noac-10ep-w${WORKERS}}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_singleprobe_1s_legacy_rgbonly_predictorlora_vitl_fp32_bs8_noac_fulltrain.yaml}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vitl.pt}"
SLURM_TIME="${SLURM_TIME:-48:00:00}"

export_csv="ALL"
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
export_csv+=",EVAL_MAX_TRAIN_ITERS=0"
export_csv+=",EVAL_NUM_EPOCHS=10"
export_csv+=",EVAL_BEST_METRIC=val-action-acc"
export_csv+=",EVAL_EARLY_STOPPING_PATIENCE=0"
export_csv+=",RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-0}"
export_csv+=",EVAL_SINGLE_PROBE=1"
export_csv+=",LORA_PROBE_TRAIN_MODE=full"
export_csv+=",EVAL_LR=0.0001"
export_csv+=",EVAL_BATCH_SIZE=8"
export_csv+=",EVAL_GRAD_CLIP=1.0"
export_csv+=",EVAL_WARMUP_EPOCHS=2"
export_csv+=",GAZE_MODE=none"
export_csv+=",LORA_CLASS_SPACE=train_only"
export_csv+=",LORA_TEMPORAL_SAMPLING=legacy"
export_csv+=",ENCODER_LORA_ENABLED=0"
export_csv+=",PREDICTOR_LORA_ENABLED=1"
export_csv+=",PREDICTOR_LORA_RANK=8"
export_csv+=",PREDICTOR_LORA_ALPHA=16.0"
export_csv+=",PREDICTOR_LORA_LAST_N_BLOCKS=0"
export_csv+=",PREDICTOR_LORA_LR_MULT=0.5"
export_csv+=",PREDICTOR_LORA_TARGET_SUFFIXES=attn.qkv|attn.proj"
export_csv+=",PREDICTOR_LORA_ACTIVATION_CHECKPOINTING=0"
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

echo "[submit-vitl-1s-legacy-rgbonly-predictorlora-fulltrain] tag=${TAG} GAZE_MODE=none, ENCODER_LORA_ENABLED=0, PREDICTOR_LORA_ENABLED=1 (rank=8 alpha=16 lr_mult=0.5 targets=attn.qkv|attn.proj)"
echo "[submit-vitl-1s-legacy-rgbonly-predictorlora-fulltrain] direct comparison target: legacy RGB-only encoder-LoRA baseline 10910499 (same split/optimization/resources, only LoRA target differs)"
echo "[submit-vitl-1s-legacy-rgbonly-predictorlora-fulltrain] partition=${PARTITION} gres=${GRES} mem=${MEM} cpus-per-task=${CPUS_PER_TASK} workers=${WORKERS}/${VAL_WORKERS}"
sbatch --time="${SLURM_TIME}" --partition="${PARTITION}" --gres="${GRES}" --mem="${MEM}" --cpus-per-task="${CPUS_PER_TASK}" --export="${export_csv}" "${RUN_SCRIPT}"
