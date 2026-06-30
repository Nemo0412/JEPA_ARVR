#!/bin/bash
#SBATCH --job-name=b12_qwen_prune_cache
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=10:00:00
#SBATCH --partition=h100_tandon
#SBATCH --output=/path/to/VJEPA2-EXP/logs/b12_qwen_prune_cache_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/b12_qwen_prune_cache_%j.err

# One-time build of the pruned post-vision-tower embedding cache for the B12 1-min->4s-token
# Qwen study. Runs the frozen vision tower + attention-importance pruning (keep_ratio=0.0667,
# ~1296 of 19440 video tokens = the 4s-level budget) and saves pruned inputs_embeds +
# position_ids per sample. Training then skips the vision tower (61% of step time) entirely.
# ~2.2 s/sample (vision-tower bound); 5202+682+1500 ≈ 7384 samples → ~5 h.

set -euo pipefail

SHARED="/path/to/VJEPA2-EXP"
WORKTREE="${SHARED}//path/to/VJEPA2-EXP"
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVERLAY="${SHARED}/overlay-15GB-500K.ext3"
ENV_NAME="EgoLifeExp"
ANN_ROOT="${SHARED}/data/hdepic_vjepa_annotations/phd_split"
CLASS_ROOT="${SHARED}/data/hd-epic-annotations/narrations-and-action-segments"

KEEP_RATIO="${KEEP_RATIO:-0.0667}"   # 1296/19440 = 4s-level token budget
OUT_DIR="${SHARED}/data/qwen_prune_embed_cache/kr${KEEP_RATIO//./p}_nf480_px256"
SPLITS="${SPLITS:-train,val,test}"

echo "[$(date)] Building pruned-embeds cache → ${OUT_DIR} (keep_ratio=${KEEP_RATIO}, splits=${SPLITS})"
mkdir -p "${OUT_DIR}" "${SHARED}/logs"

# Background GPU sampler — diagnoses the low-util watchdog cancellations.
PERF_PREFIX="${SHARED}/logs/perf_${SLURM_JOB_ID}"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 5 > "${PERF_PREFIX}_nvidia_smi.csv" 2> "${PERF_PREFIX}_nvidia_smi.err" &
PERF_PID=$!
trap 'kill ${PERF_PID} 2>/dev/null || true' EXIT

payload=""
read -r -d '' payload <<BASH || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${WORKTREE}:${SHARED}:${SHARED}/vjepa2:\${PYTHONPATH:-}"
cd "${WORKTREE}"

python scripts/build_qwen_prune_cache.py \\
  --train-csv "${ANN_ROOT}/HD_EPIC_train_vjepa.csv" \\
  --val-csv   "${ANN_ROOT}/HD_EPIC_val_vjepa.csv" \\
  --test-csv  "${ANN_ROOT}/HD_EPIC_test_vjepa.csv" \\
  --video-root "${SHARED}/data/hdepic_vjepa_videos" \\
  --verb-classes-csv "${CLASS_ROOT}/HD_EPIC_verb_classes.csv" \\
  --noun-classes-csv "${CLASS_ROOT}/HD_EPIC_noun_classes.csv" \\
  --out-dir "${OUT_DIR}" \\
  --keep-ratio ${KEEP_RATIO} \\
  --prune-chunk-size 0 \\
  --num-frames 480 \\
  --probe-num-frames 480 \\
  --target-fps 8.0 \\
  --qwen-frame-size 256 \\
  --num-workers 10 \\
  --splits "${SPLITS}"
BASH

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"

echo "[$(date)] Done. Cache dir size: $(du -sh ${OUT_DIR} | cut -f1)"
