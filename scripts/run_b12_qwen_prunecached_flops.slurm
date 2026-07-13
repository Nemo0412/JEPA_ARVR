#!/bin/bash
#SBATCH --job-name=b12_prunecached_flops
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=01:00:00
#SBATCH --partition=h100_tandon
#SBATCH --output=/path/to/VJEPA2-EXP/logs/b12_prunecached_flops_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/b12_prunecached_flops_%j.err

# Measure FLOPs for the cached pruned-Qwen (1min->4s-token) method:
#   (A) one-time cache build (ViT 480f + prune), (B) per-epoch cached LLM train step.
set -euo pipefail
SHARED="/path/to/VJEPA2-EXP"
WORKTREE="${SHARED}//path/to/VJEPA2-EXP"
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVERLAY="${SHARED}/overlay-15GB-500K.ext3"

payload=""
read -r -d '' payload <<BASH || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/EgoLifeExp
export PYTHONPATH="${WORKTREE}:${SHARED}:${SHARED}/vjepa2:\${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${WORKTREE}"
# Single forward through the FULL model with in-model pruning at 1min: ViT(480f)→prune→LLM(1297).
# Directly comparable to the no-prune forward FLOPs (4s=15.45, 1min=333.0).
python scripts/profile_b12_1min_flops.py --windows 480 --keep-ratio 0.0667 --n-runs 2
BASH

singularity exec --nv --overlay "${OVERLAY}:ro" "${SIF}" /bin/bash -lc "${payload}"
