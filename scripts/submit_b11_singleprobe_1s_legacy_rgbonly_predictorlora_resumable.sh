#!/bin/bash
set -euo pipefail

# Resumable variant of submit_b11_singleprobe_1s_legacy_rgbonly_predictorlora_vitl_fp32_bs8_noac_fulltrain.sh
# for unattended relaunch by watch_and_relaunch_predictorlora.sh.
#
# This run has been repeatedly killed by cluster-side GPU quota contention --
# external SIGNAL Terminated cancellations with Reason=QOSMaxGRESPerUser
# (this account's own concurrent jobs) or QOSGrpGRES (shared group-level GPU
# quota), not OOM and not a code bug (gradients/accuracy stayed healthy across
# every attempt: 10999272 OOM'd at MEM=256GB/itr620; 11010450/11043844/11076126/
# 11086217/11094405 all externally cancelled at MEM=512GB between itr~90 and
# full-epoch-2 depending on cluster load at the time). Predictor-LoRA checkpoint
# persistence was added (predictor_lora_latest.pt written on every latest.pt
# save) so each relaunch can resume from the last completed epoch instead of
# restarting from epoch 0.
#
# Always sets RESUME_CHECKPOINT=1: harmless on a fresh run (no latest.pt yet),
# required for relaunches to continue.
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
export_csv+=",RESUME_CHECKPOINT=1"
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
MEM="${SLURM_MEM:-512GB}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-16}"

echo "[submit-vitl-1s-legacy-rgbonly-predictorlora-resumable] tag=${TAG} resume=1 workers=${WORKERS}/${VAL_WORKERS}"
echo "[submit-vitl-1s-legacy-rgbonly-predictorlora-resumable] partition=${PARTITION} gres=${GRES} mem=${MEM} cpus-per-task=${CPUS_PER_TASK}"
sbatch --time="${SLURM_TIME}" --partition="${PARTITION}" --gres="${GRES}" --mem="${MEM}" --cpus-per-task="${CPUS_PER_TASK}" --export="${export_csv}" "${RUN_SCRIPT}"
