#!/bin/bash
set -euo pipefail

# B12 1-minute window feasibility smoke: Qwen2.5-VL-3B receiving a full 1-min observation
# window (480 frames @ 8fps, 256px, keep_ratio=1.0 / no pruning).
#
# PURPOSE: measure raw pipeline timing WITHOUT preprocessing cache.  If the GPU utilisation
# is low (<60%), the pipeline is data-bound and we should add --preproc-cache-dir (same
# mechanism as the 4s runs).
#
# WINDOW: 60s * 8fps = 480 frames → 240 temporal slots (temporal_patch_size=2) × 81 spatial
# patches (18x18 / spatial_merge_size=2) = 19 440 merged video tokens at keep=1.0.
# Total seq len ≈ 19 440 + ~100 text < 32 768 (Qwen max_position_embeddings), so M-RoPE
# positions are valid.
#
# OOM guard: BATCH_SIZE=1 (19 440 tokens × 36-layer GQA KV ≈ 13 GB alone); GRAD_ACCUM=8
# keeps effective batch = 8, matching prior runs' training dynamics.
#
# If smoke confirms feasibility → run full training (10 epochs) with cache.
#
# Usage:
#   bash scripts/submit_b12_qwen_1min_raw_smoke.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SCRIPT_DIR}/submit_vlm_probe_lora.sh"
SHARED="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
ANN_ROOT="${SHARED}/data/hdepic_vjepa_annotations/phd_split"

export BACKEND="qwen25vl"
export TRAIN_CSV="${ANN_ROOT}/HD_EPIC_train_vjepa.csv"
export VAL_CSV="${ANN_ROOT}/HD_EPIC_val_vjepa.csv"
export TEST_CSV="${ANN_ROOT}/HD_EPIC_test_vjepa.csv"

# 1-minute window
export NUM_FRAMES="480"
export TARGET_FPS="8.0"
export PROBE_NUM_FRAMES="480"    # no temporal downsampling (all 480 frames fed)
export QWEN_FRAME_SIZE="256"

# No pruning — raw anchor
export PRUNE_KEEP_RATIO="1.0"
export PRUNE_CHUNK_SIZE="0"

# No cache: measure raw decode+tokenize throughput
export PREPROC_CACHE_DIR=""

# OOM guard: 19 440 tokens at batch>1 likely OOMs on H100 80 GB
export BATCH_SIZE="1"
export GRAD_ACCUM_STEPS="8"    # effective batch = 8, matching prior runs

# Smoke: 32 samples, 1 epoch, log every step for per-batch timing
export MAX_TRAIN_SAMPLES="32"
export NUM_EPOCHS="1"
export LOG_EVERY="1"

export NUM_WORKERS="10"
export SLURM_CPUS_PER_TASK="12"
export SLURM_PARTITION="h100_tandon"
export SLURM_GRES="gpu:h100:1"
export SLURM_TIME="02:00:00"
export LOCAL_FILES_ONLY="1"

export RUN_TAG="b12_qwen_1min_raw_smoke"

RUN_TAG="${RUN_TAG}" bash "${SUBMIT}"
