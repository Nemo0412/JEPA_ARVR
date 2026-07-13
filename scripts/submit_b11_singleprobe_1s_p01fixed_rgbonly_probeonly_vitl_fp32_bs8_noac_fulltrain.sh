#!/bin/bash
set -euo pipefail

# ViT-L @256 fp32 fulltrain, 1s concat_ar, p01_fixed split, RGB-only
# (no gaze, no pose), probe-only (encoder fully frozen, no encoder LoRA).
# This is the "clean baseline" probe referenced in the latent-memory-prelim
# LTM work (S0 plan): no input modality besides RGB, no context/history
# injection, classifier head covers the full HD-EPIC taxonomy via the
# project's current default class_space=phd_reference / temporal_sampling=
# phd_reference (post commit 8f1adea) -- intentionally NOT overridden to
# train_only/legacy here, since the p01_fixed CSVs (unlike the legacy ones)
# satisfy the stricter phd_reference action-coverage check.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
# run_hdepic_lora_probe.slurm regenerates the CONFIG_PATH yaml from a template on every
# launch and rewrites dataset_train/val/test from ADAPTED_ANNOTATIONS_DIR -- any direct
# edits to the generated yaml's CSV paths get clobbered. Must override via this env var.
ADAPTED_ANNOTATIONS_DIR="${ADAPTED_ANNOTATIONS_DIR:-${PROJECT_ROOT}/data/hdepic_vjepa_annotations/p01_fixed}"
# _hdepic_reference_paths() in eval.py derives the PhD-reference class-space root as
# dataset_train.parent.parent, which assumes dataset CSVs sit exactly one directory
# under data/. The p01_fixed subdir adds an extra level, breaking that heuristic, so
# point these explicitly at the real shared location under data/hd-epic-annotations/.
HDEPIC_ANN_ROOT="${HDEPIC_ANN_ROOT:-${PROJECT_ROOT}/data/hd-epic-annotations/narrations-and-action-segments}"
WORKERS="${EVAL_NUM_WORKERS:-10}"
VAL_WORKERS="${EVAL_VAL_NUM_WORKERS:-${WORKERS}}"
TAG="${LORA_TAG:-hdepic-singleprobe-1s-p01fixed-rgbonly-probeonly-vitl-fp32-bs8-noac-10ep-w${WORKERS}}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_singleprobe_1s_p01fixed_rgbonly_probeonly_vitl_fp32_bs8_noac_fulltrain.yaml}"
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
export_csv+=",RESUME_CHECKPOINT=0"
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

PARTITION="${SLURM_PARTITION:-h100_tandon}"
GRES="${SLURM_GRES:-gpu:h100:1}"
# 10905928 (4 workers) peaked at ~83.5GB cgroup mem against a 384GB request.
# Scaling workers 4 -> 10 with a ~16GB/worker margin: ~20GB base + 10*16 ~ 180GB; round up for headroom.
MEM="${SLURM_MEM:-256GB}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-16}"

echo "[submit-vitl-1s-p01fixed-rgbonly-probeonly-fulltrain] tag=${TAG}"
echo "[submit-vitl-1s-p01fixed-rgbonly-probeonly-fulltrain] split=p01_fixed (5978/376/1046 rows); GAZE_MODE=none, ENCODER_LORA_ENABLED=0 (encoder frozen, probe-only); class_space/temporal_sampling left at code default (phd_reference)"
echo "[submit-vitl-1s-p01fixed-rgbonly-probeonly-fulltrain] partition=${PARTITION} gres=${GRES} mem=${MEM} cpus-per-task=${CPUS_PER_TASK} workers=${WORKERS}/${VAL_WORKERS}"
sbatch --time="${SLURM_TIME}" --partition="${PARTITION}" --gres="${GRES}" --mem="${MEM}" --cpus-per-task="${CPUS_PER_TASK}" --export="${export_csv}" "${RUN_SCRIPT}"
