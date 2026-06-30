#!/bin/bash
#SBATCH --job-name=b12_qwen_prune_4s_train
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --partition=h100_tandon
#SBATCH --output=/path/to/VJEPA2-EXP/logs/b12_qwen_prune_4s_train_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/b12_qwen_prune_4s_train_%j.err

# B12 1-min->4s-token Qwen training on the pre-pruned embedding cache.
# Vision tower is skipped entirely (cached); only the LLM (LoRA) + 3 heads train, on the
# ~1296 surviving video tokens per sample. LLM-only step ≈ 0.16 s/sample → ~15 min/epoch.
#
# Env knobs:
#   KEEP_RATIO   (default 0.0667)  -- selects the cache dir
#   TRAIN_CSV / VAL_CSV / TEST_CSV -- override for smoke (default full phd_split)
#   MAX_TRAIN_SAMPLES (default 0)  -- smoke cap
#   NUM_EPOCHS   (default 10)
#   RUN_TAG      (default b12_qwen_prune4s_kr<kr>)

set -euo pipefail

SHARED="/path/to/VJEPA2-EXP"
WORKTREE="${SHARED}//path/to/VJEPA2-EXP"
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVERLAY="${SHARED}/overlay-15GB-500K.ext3"
ENV_NAME="EgoLifeExp"
ANN_ROOT="${SHARED}/data/hdepic_vjepa_annotations/phd_split"
CLASS_ROOT="${SHARED}/data/hd-epic-annotations/narrations-and-action-segments"

KEEP_RATIO="${KEEP_RATIO:-0.0667}"
CACHE_DIR="${SHARED}/data/qwen_prune_embed_cache/kr${KEEP_RATIO//./p}_nf480_px256"
TRAIN_CSV="${TRAIN_CSV:-${ANN_ROOT}/HD_EPIC_train_vjepa.csv}"
VAL_CSV="${VAL_CSV:-${ANN_ROOT}/HD_EPIC_val_vjepa.csv}"
TEST_CSV="${TEST_CSV:-${ANN_ROOT}/HD_EPIC_test_vjepa.csv}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
# LLM-only step on ~1296 tokens is tiny; batch up so the GPU is well-fed (avoids low-util
# cancellation) and fast. eff batch = BATCH_SIZE × GRAD_ACCUM = 8 (matches the 1min run).
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
RUN_TAG="${RUN_TAG:-b12_qwen_prune4s_kr${KEEP_RATIO//./p}}"
OUTPUT_DIR="${SHARED}/outputs/vlm_probe_lora/${RUN_TAG}"
RESUME_FROM="${RESUME_FROM:-}"   # set to probe-last.pt to continue a preempted run

echo "[$(date)] Train from cache ${CACHE_DIR} → ${OUTPUT_DIR} (epochs=${NUM_EPOCHS} max_train=${MAX_TRAIN_SAMPLES})"
mkdir -p "${OUTPUT_DIR}" "${SHARED}/logs"

payload=""
read -r -d '' payload <<BASH || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${WORKTREE}:${SHARED}:${SHARED}/vjepa2:\${PYTHONPATH:-}"
cd "${WORKTREE}"

python -m app.hdepic_lora_action_anticipation.train_vlm_probe_lora \\
  --backend qwen25vl \\
  --train-csv "${TRAIN_CSV}" \\
  --val-csv   "${VAL_CSV}" \\
  --test-csv  "${TEST_CSV}" \\
  --video-root "${SHARED}/data/hdepic_vjepa_videos" \\
  --verb-classes-csv "${CLASS_ROOT}/HD_EPIC_verb_classes.csv" \\
  --noun-classes-csv "${CLASS_ROOT}/HD_EPIC_noun_classes.csv" \\
  --output-dir "${OUTPUT_DIR}" \\
  --cached-embeds-dir "${CACHE_DIR}" \\
  --num-frames 480 --probe-num-frames 480 --target-fps 8.0 --qwen-frame-size 256 \\
  --prune-keep-ratio 1.0 \\
  --max-train-samples ${MAX_TRAIN_SAMPLES} \\
  --num-epochs ${NUM_EPOCHS} \\
  --lr 2e-4 --weight-decay 1e-4 --warmup-epochs 2 \\
  --lora-rank 16 --lora-alpha 32.0 \\
  --batch-size ${BATCH_SIZE} --grad-accum-steps ${GRAD_ACCUM_STEPS} \\
  --log-every 20 --torch-dtype bfloat16 --num-workers 8 --seed 0 \\
  ${RESUME_FROM:+--resume-from "${RESUME_FROM}"} \\
  --local-files-only
BASH

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
