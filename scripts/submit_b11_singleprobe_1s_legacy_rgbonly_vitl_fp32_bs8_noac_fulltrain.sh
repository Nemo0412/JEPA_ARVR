#!/bin/bash
set -euo pipefail

# ViT-L @256 fp32 fulltrain, 1s concat_ar, RGB-only (no gaze, no pose) baseline.
# Matched to the two recent legacy-split optimization runs so all three are
# directly comparable:
#   - RGB+gaze         (job 10847438, tag hdepic-singleprobe-1s-legacy-gaze-nopose-...)
#   - RGB+gaze+pose    (job 10843819, tag hdepic-singleprobe-1s-p01fixed-regcp-...)
# This run mirrors the RGB+gaze config exactly (no confidence_penalty,
# early_stopping_patience=0) since that's the cleaner of the two references;
# the gaze+pose run additionally enables confidence_penalty=0.01 and
# early_stopping_patience=3, which are NOT part of this baseline's settings.
#
# Split note: both reference runs read from the generic
# HD_EPIC_{train,val,test}_vjepa.csv paths, whose on-disk split fingerprint
# drifted between submissions (RGB+gaze legacy split (5744/1397) was a different
# materialized state than RGB+gaze+pose's mislabeled 5964/319). Neither is the
# project's current `p01_fixed` standard (5964/375/1045). This run uses
# whatever split is currently materialized at those CSV paths -- confirm it
# matches the RGB+gaze run's legacy fingerprint before trusting a 3-way
# comparison; see docs/HD_EPIC_SYNC_NOTES.md.
#
# Code-default note: 10847438 ran before commit 8f1adea (2026-06-15 10:24)
# switched the project default to class_space=phd_reference and
# temporal_sampling=phd_reference. Those strict defaults reject this legacy
# CSV's action coverage (see docs/HD_EPIC_SYNC_NOTES.md). This
# run explicitly sets LORA_CLASS_SPACE=train_only and
# LORA_TEMPORAL_SAMPLING=legacy to reproduce 10847438's actual runtime
# behavior (job 10904424 failed without these).

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
WORKERS="${EVAL_NUM_WORKERS:-10}"
VAL_WORKERS="${EVAL_VAL_NUM_WORKERS:-${WORKERS}}"
TAG="${LORA_TAG:-hdepic-singleprobe-1s-legacy-rgbonly-vitl-fp32-bs8-noac-10ep-w${WORKERS}}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_singleprobe_1s_legacy_rgbonly_vitl_fp32_bs8_noac_fulltrain.yaml}"
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
export_csv+=",RESUME_CHECKPOINT=0"
export_csv+=",EVAL_SINGLE_PROBE=1"
export_csv+=",LORA_PROBE_TRAIN_MODE=full"
export_csv+=",EVAL_LR=0.0001"
export_csv+=",EVAL_BATCH_SIZE=8"
export_csv+=",EVAL_GRAD_CLIP=1.0"
export_csv+=",EVAL_WARMUP_EPOCHS=2"
export_csv+=",GAZE_MODE=none"
export_csv+=",LORA_CLASS_SPACE=train_only"
export_csv+=",LORA_TEMPORAL_SAMPLING=legacy"
export_csv+=",ENCODER_LORA_ENABLED=1"
export_csv+=",ENCODER_LORA_RANK=8"
export_csv+=",ENCODER_LORA_ALPHA=16.0"
export_csv+=",ENCODER_LORA_LAST_N_BLOCKS=0"
export_csv+=",ENCODER_LORA_LR_MULT=0.5"
export_csv+=",ENCODER_LORA_TARGET_SUFFIXES=attn.qkv|attn.proj"
export_csv+=",ENCODER_LORA_ACTIVATION_CHECKPOINTING=0"
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
# 10905928 (this same job, 4 workers) was killed by the cluster's low-GPU-utilization
# watchdog after exactly 2h (per-step `data:` time ~6.1s dominating GPU compute time);
# cgroup mem peaked at ~83.5GB against a 384GB allocation. Raise workers to relieve the
# dataloader bottleneck; size mem from the observed peak (~20GB base + ~16GB/worker)
# instead of carrying over the old 768GB default.
MEM="${SLURM_MEM:-256GB}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-16}"

echo "[submit-vitl-1s-legacy-rgbonly-fulltrain] tag=${TAG} GAZE_MODE=none (RGB-only baseline)"
echo "[submit-vitl-1s-legacy-rgbonly-fulltrain] matched to RGB+gaze job 10847438; split=whatever is materialized at HD_EPIC_*_vjepa.csv now (verify legacy 5744/1397 fingerprint in the run log)"
echo "[submit-vitl-1s-legacy-rgbonly-fulltrain] partition=${PARTITION} gres=${GRES} mem=${MEM} cpus-per-task=${CPUS_PER_TASK} workers=${WORKERS}/${VAL_WORKERS} (10904931 OOM'd on a 44GB GPU from the generic run_hdepic_lora_probe.slurm defaults; pin H100 to match the references; 10905928 low-GPU-util killed at 4 workers)"
sbatch --time="${SLURM_TIME}" --partition="${PARTITION}" --gres="${GRES}" --mem="${MEM}" --cpus-per-task="${CPUS_PER_TASK}" --export="${export_csv}" "${RUN_SCRIPT}"
