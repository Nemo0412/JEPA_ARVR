#!/bin/bash
set -euo pipefail

# B12 1-minute Qwen training resume from epoch 1 checkpoint.
#
# Why: job 11831042 was killed during epoch-0 val due to low GPU util (val is decode-bound,
#      GPU idle while 682 val samples decode at ~2.4s each). Fix: enable val preprocessing
#      cache (PREPROC_CACHE_DIR set). Train is GPU-bound (no cache needed for train; train
#      samples will just miss the cache and compute on-the-fly, which is fine). Val samples
#      find their pre-built cache files → val drops from 31 min → ~5 min, GPU util stays
#      above 60% for the full 2-hour monitoring window.
#
# Prerequisites:
#   1. Val cache built: sbatch scripts/run_b12_qwen_1min_val_cache.slurm
#   2. Checkpoint exists: outputs/vlm_probe_lora/b12_qwen_1min_fulltrain_kr1p0/probe-last.pt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SCRIPT_DIR}/submit_vlm_probe_lora.sh"
SHARED="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
ANN_ROOT="${SHARED}/data/hdepic_vjepa_annotations/phd_split"

CKPT="${SHARED}/outputs/vlm_probe_lora/b12_qwen_1min_fulltrain_kr1p0/probe-last.pt"
if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  exit 1
fi

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

# Val cache pre-built at this path (train samples will miss and compute on-the-fly, fine).
export PREPROC_CACHE_DIR="${SHARED}/data/preproc_cache_qwen"

export BATCH_SIZE="1"
export GRAD_ACCUM_STEPS="8"
export NUM_WORKERS="10"
export SLURM_CPUS_PER_TASK="12"

export MAX_TRAIN_SAMPLES="0"   # full training set
export NUM_EPOCHS="10"
export LOG_EVERY="20"

export RESUME_FROM="${CKPT}"   # picked up by submit_vlm_probe_lora.sh → --resume-from flag

export SLURM_PARTITION="h100_tandon"
export SLURM_GRES="gpu:h100:1"
export SLURM_TIME="24:00:00"
export LOCAL_FILES_ONLY="1"

# Same output dir as original run so probe-last.pt/probe-best.pt are shared
export RUN_TAG="b12_qwen_1min_fulltrain_kr1p0"

echo "[submit_b12_qwen_1min_resume] Resuming from ${CKPT}"
echo "[submit_b12_qwen_1min_resume] Val cache dir: ${PREPROC_CACHE_DIR}"
RUN_TAG="${RUN_TAG}" bash "${SUBMIT}"
