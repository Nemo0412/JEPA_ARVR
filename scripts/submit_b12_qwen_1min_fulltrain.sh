#!/bin/bash
set -euo pipefail

# B12 1-minute window full training: Qwen2.5-VL-3B on 60s observation window.
#
# Key settings:
#   - 480 frames @ 8fps @ 256px, keep_ratio=1.0 (no pruning) — full 1-min context
#   - phd_split (train 5214 / val 682 / test 1501)
#   - BATCH_SIZE=1 / GRAD_ACCUM=8 (eff batch=8; 19440 LLM tokens → OOM at larger batch)
#   - NO preprocessing cache: GPU/sample (3.2s) > decode/sample (~2.4s) → GPU-bound,
#     10 workers keep it fed (opposite of 4s runs which were data-bound)
#   - Time budget: ~13h (46 min train + 30 min val per epoch × 10 epochs); request 24h

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SCRIPT_DIR}/submit_vlm_probe_lora.sh"
SHARED="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
ANN_ROOT="${SHARED}/data/hdepic_vjepa_annotations/phd_split"

export BACKEND="qwen25vl"
export TRAIN_CSV="${ANN_ROOT}/HD_EPIC_train_vjepa.csv"
export VAL_CSV="${ANN_ROOT}/HD_EPIC_val_vjepa.csv"
export TEST_CSV="${ANN_ROOT}/HD_EPIC_test_vjepa.csv"

export NUM_FRAMES="480"
export TARGET_FPS="8.0"
export PROBE_NUM_FRAMES="480"
export QWEN_FRAME_SIZE="256"

export PRUNE_KEEP_RATIO="1.0"
export PRUNE_CHUNK_SIZE="0"
export PREPROC_CACHE_DIR=""   # no cache: GPU-bound at this window length

export BATCH_SIZE="1"
export GRAD_ACCUM_STEPS="8"
export NUM_WORKERS="10"
export SLURM_CPUS_PER_TASK="12"

export MAX_TRAIN_SAMPLES="0"   # full training set
export NUM_EPOCHS="10"
export LOG_EVERY="20"

export SLURM_PARTITION="h100_tandon"
export SLURM_GRES="gpu:h100:1"
export SLURM_TIME="24:00:00"
export LOCAL_FILES_ONLY="1"

export RUN_TAG="b12_qwen_1min_fulltrain_kr1p0"

RUN_TAG="${RUN_TAG}" bash "${SUBMIT}"
