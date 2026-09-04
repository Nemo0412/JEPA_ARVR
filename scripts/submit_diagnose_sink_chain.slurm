#!/bin/bash
#SBATCH --job-name=sink_chain
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:h100:1 --cpus-per-task=8 --mem=80GB --time=00:25:00
#SBATCH --output=/scratch/ll5914/logs/sink_chain_%j.out
#SBATCH --error=/scratch/ll5914/logs/sink_chain_%j.err

set -euo pipefail
export PYTHONPATH="/home/ll5914/Jepa_yifan/JEPA_ARVR:/home/ll5914/ARVR_Video/vjepa2:${PYTHONPATH:-}"
export VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
export TRI_MODAL_FRAME_CACHE="/scratch/ll5914/datasets/HD-EPIC/_clip_frame_cache/p01_f32_at1"
cd /home/ll5914/Jepa_yifan/JEPA_ARVR
/scratch/ll5914/conda_envs/SVD/bin/python scripts/diagnose_sink_attention_chain.py
