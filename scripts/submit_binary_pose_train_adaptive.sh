#!/usr/bin/env bash
set -euo pipefail

# Adaptive full P01 binary+pose train submitter.
# Default: w10 + 768G (H100 4-GPU node ~1.5TB; 768G balances RAM headroom vs schedulability).

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
retry="${RELAUNCH_COUNT:-0}"
reason_raw="${FAIL_REASON:-${CANCEL_REASON:-}}"
reason_u="$(echo "${reason_raw}" | tr '[:lower:]' '[:upper:]')"

is_low_usage=0
is_oom=0
if [[ "${reason_u}" == *GPU* || "${reason_u}" == *USAGE* || "${reason_u}" == *UTIL* ]]; then
  is_low_usage=1
fi
if [[ "${reason_u}" == *OOM* || "${reason_u}" == *OUT_OF_MEM* || "${reason_u}" == *KILL* || "${reason_u}" == *SIGKILL* ]]; then
  is_oom=1
fi

export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
export DEBUG_SUBSET_PATH=""
export EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-5}"
export PERF_MONITOR=1
export GAZE_DECODER_CACHE_READER=1

# Slurm mem (override via SLURM_MEM=896G etc.)
export SLURM_MEM="${SLURM_MEM:-768G}"

if (( is_low_usage == 1 && is_oom == 0 )); then
  export SLURM_MEM="${SLURM_MEM:-768G}"
  case "${retry}" in
    0)
      export EVAL_NUM_WORKERS=10
      export GAZE_DECODER_NUM_THREADS=3
      export EVAL_VAL_NUM_WORKERS=2
      export LORA_TAG="hdepic-lora-binary-gaze-pose-rnn-5ep-bs2-w10-m768-lowutil-r1-p01"
      ;;
    1)
      export EVAL_NUM_WORKERS=10
      export GAZE_DECODER_NUM_THREADS=3
      export EVAL_VAL_NUM_WORKERS=0
      export LORA_TAG="hdepic-lora-binary-gaze-pose-rnn-5ep-bs2-w10-m768-lowutil-r2-p01"
      ;;
    *)
      export EVAL_NUM_WORKERS=10
      export GAZE_DECODER_NUM_THREADS=2
      export EVAL_VAL_NUM_WORKERS=0
      export LORA_TAG="hdepic-lora-binary-gaze-pose-rnn-5ep-bs2-w10-m768-lowutil-r3-p01"
      ;;
  esac
elif (( is_oom == 1 )); then
  case "${retry}" in
    0)
      export SLURM_MEM="${SLURM_MEM_OOM:-768G}"
      export EVAL_NUM_WORKERS=8
      export GAZE_DECODER_NUM_THREADS=2
      export EVAL_VAL_NUM_WORKERS=2
      export LORA_TAG="hdepic-lora-binary-gaze-pose-rnn-5ep-bs2-w8-m768-oom-r1-p01"
      ;;
    1)
      export SLURM_MEM="${SLURM_MEM_OOM:-640G}"
      export EVAL_NUM_WORKERS=6
      export GAZE_DECODER_NUM_THREADS=2
      export EVAL_VAL_NUM_WORKERS=2
      export LORA_TAG="hdepic-lora-binary-gaze-pose-rnn-5ep-bs2-w6-m640-oom-r2-p01"
      ;;
    *)
      export SLURM_MEM="${SLURM_MEM_OOM:-640G}"
      export EVAL_NUM_WORKERS=4
      export GAZE_DECODER_NUM_THREADS=1
      export EVAL_VAL_NUM_WORKERS=2
      export LORA_TAG="hdepic-lora-binary-gaze-pose-rnn-5ep-bs2-w4-m640-oom-r3-p01"
      ;;
  esac
else
  export EVAL_NUM_WORKERS=10
  export GAZE_DECODER_NUM_THREADS=2
  export EVAL_VAL_NUM_WORKERS=2
  export LORA_TAG="${LORA_TAG:-hdepic-lora-binary-gaze-pose-rnn-5ep-bs2-w10-m768-p01}"
fi

echo "[adaptive-train] RELAUNCH_COUNT=${retry} mem=${SLURM_MEM} oom=${is_oom} low_usage=${is_low_usage} reason='${reason_raw}' workers=${EVAL_NUM_WORKERS} tag=${LORA_TAG}"
sbatch --mem="${SLURM_MEM}" "${PROJECT_ROOT}/scripts/run_hdepic_lora_binary_pose_train.slurm"
