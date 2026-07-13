#!/bin/bash
set -euo pipefail

# Resumable variant of submit_b11_singleprobe_1s_p01fixed_rgbonly_probeonly_vitl_fp32_bs8_noac_fulltrain.sh
# for unattended relaunch by watch_and_relaunch_p01fixed_rgbonly_probeonly.sh.
#
# This baseline is probe-only (encoder frozen, ENCODER_LORA_ENABLED=0): GPU compute per
# step is tiny, so it cannot hide the dataloader's ~3.5-4s/batch decode latency the way
# the encoder-LoRA runs do (their longer backward pass overlaps with prefetch and keeps
# average GPU utilization above the cluster's low-usage watchdog threshold). Even at
# num_workers=10 this run was killed twice at ~2h12m by that watchdog (jobs 10912829,
# 10912919). batch_size is intentionally left at 8 to stay comparable with the
# RGB+gaze / RGB+gaze+pose baselines -- do not bump it to fix this; use resume+relaunch
# instead. See docs/HD_EPIC_SYNC_NOTES.md for the split contract.
#
# Always sets RESUME_CHECKPOINT=1: harmless on a fresh run (no latest.pt yet), and
# required for relaunches to continue from the last completed epoch instead of
# restarting from epoch 0.
#
# WORKERS history (2026-06-16): w14 against MEM=256GB caused real cgroup OOM
# three times in a row (jobs 10918458, 10922368, 10927101; mem climbed to the
# 256GB ceiling within 22min-1h37m each time). Reverted to w10/256GB, which
# ran cleanly for 2h12m+ in earlier attempts before being killed by the
# (harmless, auto-relaunchable) low-GPU-util watchdog instead. Now raising
# MEM to 640GB (H100 nodes have 1.5TB/96 CPUs; this is comfortably available)
# to give real headroom for w16, rather than fighting OOM by capping workers.
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
ADAPTED_ANNOTATIONS_DIR="${ADAPTED_ANNOTATIONS_DIR:-${PROJECT_ROOT}/data/hdepic_vjepa_annotations/p01_fixed}"
HDEPIC_ANN_ROOT="${HDEPIC_ANN_ROOT:-${PROJECT_ROOT}/data/hd-epic-annotations/narrations-and-action-segments}"
WORKERS="${EVAL_NUM_WORKERS:-16}"
VAL_WORKERS="${EVAL_VAL_NUM_WORKERS:-${WORKERS}}"
TAG="${LORA_TAG:-hdepic-singleprobe-1s-p01fixed-rgbonly-probeonly-vitl-fp32-bs8-noac-10ep-w10}"
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

PARTITION="${SLURM_PARTITION:-h100_tandon}"
GRES="${SLURM_GRES:-gpu:h100:1}"
MEM="${SLURM_MEM:-640GB}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-24}"

echo "[submit-vitl-1s-p01fixed-rgbonly-probeonly-resumable] tag=${TAG} resume=1 workers=${WORKERS}/${VAL_WORKERS}"
echo "[submit-vitl-1s-p01fixed-rgbonly-probeonly-resumable] partition=${PARTITION} gres=${GRES} mem=${MEM} cpus-per-task=${CPUS_PER_TASK}"
sbatch --time="${SLURM_TIME}" --partition="${PARTITION}" --gres="${GRES}" --mem="${MEM}" --cpus-per-task="${CPUS_PER_TASK}" --export="${export_csv}" "${RUN_SCRIPT}"
