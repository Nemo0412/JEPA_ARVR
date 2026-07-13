#!/bin/bash
#SBATCH --job-name=b12_qwen_val_cache
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/b12_qwen_val_cache_%j.out
#SBATCH --error=logs/slurm/b12_qwen_val_cache_%j.err

set -euo pipefail

SHARED="/path/to/VJEPA2-EXP"
WORKTREE="${SHARED}//path/to/VJEPA2-EXP"

CACHE_DIR="${SHARED}/data/preproc_cache_qwen/qwen25vl_nf480_pnf480_fps8.0_px256"
VAL_CSV="${SHARED}/data/hdepic_vjepa_annotations/phd_split/HD_EPIC_val_vjepa.csv"
VIDEO_ROOT="${SHARED}/data/hdepic_vjepa_videos"

echo "[$(date)] Building Qwen 480f val cache → ${CACHE_DIR}"
mkdir -p "${CACHE_DIR}"
mkdir -p logs/slurm

singularity exec \
  --nv \
  --overlay "${SHARED}/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/env.sh
    conda activate /ext3/envs/EgoLifeExp
    cd ${WORKTREE}
    export PYTHONPATH=${WORKTREE}:${SHARED}:\${PYTHONPATH:-}
    python scripts/build_val_cache_1min.py \
      --val-csv  ${VAL_CSV} \
      --video-root ${VIDEO_ROOT} \
      --cache-dir ${CACHE_DIR} \
      --num-workers 14 \
      --num-frames 480 \
      --target-fps 8.0 \
      --decode-size 256
  "

echo "[$(date)] Done. Cache dir size: \$(du -sh ${CACHE_DIR} | cut -f1)"
