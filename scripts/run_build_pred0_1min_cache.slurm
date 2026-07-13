#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__pred0_cache
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/pred0_cache_%A_%a.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/pred0_cache_%A_%a.err

# Build predictor-block-0-guided pruning cache from 1min full-ctx tokens.
# Run as a 4-shard job array: sbatch --array=0-3 this_script.slurm
# Each shard processes ~1844 samples (~75 min at 2.4 s/sample).
#
# Input : data/preproc_cache_vjepa/nf480_fps8.0_px256_keep61440_idx/
# Output: data/preproc_cache_vjepa/nf480_fps8.0_px256_pred0_keep4096_idx/

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP//path/to/VJEPA2-EXP}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD="${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}"
export PROJECT_ROOT SHARED_PROJECT_ROOT ENV_NAME NUM_SHARDS SHARD

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "[pred0_cache] shard=${SHARD}/${NUM_SHARDS}"

payload=""
read -r -d '' payload <<'BASH' || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
export CACHE_SHARD="${SHARD}"
export CACHE_NUM_SHARDS="${NUM_SHARDS}"
export SHARED_PROJECT_ROOT
python scripts/build_pred0_prune_1min_cache.py
BASH

singularity exec --nv --overlay "${OVERLAY}:ro" "${SIF}" /bin/bash -lc "${payload}"
