#!/bin/bash
set -euo pipefail

# B12 7-second window: Qwen2.5-VL-3B full-context training with processor-tensor cache.
#
# 56 frames @ 8 fps @ 256 px => 28 temporal slots × 9 × 9 = 2268 Qwen video tokens.
# This is the "Qwen full" row for the 7s extension: no token pruning, cache enabled.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SCRIPT_DIR}/submit_vlm_probe_lora.sh"
SHARED="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
ANN_ROOT="${SHARED}/data/hdepic_vjepa_annotations/phd_split"

export BACKEND="qwen25vl"
export TRAIN_CSV="${ANN_ROOT}/HD_EPIC_train_vjepa.csv"
export VAL_CSV="${ANN_ROOT}/HD_EPIC_val_vjepa.csv"
export TEST_CSV="${ANN_ROOT}/HD_EPIC_test_vjepa.csv"

export NUM_FRAMES="56"
export TARGET_FPS="8.0"
export PROBE_NUM_FRAMES="56"
export QWEN_FRAME_SIZE="256"

export PRUNE_KEEP_RATIO="1.0"
export PRUNE_CHUNK_SIZE="0"
export PREPROC_CACHE_DIR="${SHARED}/data/preproc_cache_qwen"

# Edge clips can be shorter than 56 frames, producing variable Qwen sequence lengths. The normal
# processor-tensor collator concatenates tensors directly, so keep microbatch=1 and recover the
# 1min run's effective batch size through grad accumulation.
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
export NUM_WORKERS="${NUM_WORKERS:-10}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-12}"

export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export LOG_EVERY="${LOG_EVERY:-20}"

export SLURM_PARTITION="${SLURM_PARTITION:-h100_tandon}"
export SLURM_GRES="${SLURM_GRES:-gpu:h100:1}"
export SLURM_MEM="${SLURM_MEM:-128GB}"
export SLURM_TIME="${SLURM_TIME:-24:00:00}"
export LOCAL_FILES_ONLY="1"

export RUN_TAG="${RUN_TAG:-b12_qwen_7s_fullctx_kr1p0_cache}"

RUN_TAG="${RUN_TAG}" bash "${SUBMIT}"
