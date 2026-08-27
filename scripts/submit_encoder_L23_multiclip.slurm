#!/bin/bash
#SBATCH --job-name=enc_L23
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1 --cpus-per-task=8 --mem=64GB --time=00:30:00
#SBATCH --output=/scratch/ll5914/logs/enc_L23_multiclip_%j.out
#SBATCH --error=/scratch/ll5914/logs/enc_L23_multiclip_%j.err

set -euo pipefail
export PYTHONPATH="/home/ll5914/Jepa_yifan/JEPA_ARVR:/home/ll5914/ARVR_Video/vjepa2:${PYTHONPATH:-}"
export TRI_MODAL_FRAME_CACHE="/scratch/ll5914/datasets/HD-EPIC/_clip_frame_cache/p01_f32_at1"
export VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
cd /home/ll5914/Jepa_yifan/JEPA_ARVR
/scratch/ll5914/conda_envs/SVD/bin/python scripts/analyze_encoder_last_layer_multiclip.py \
  --n-sample "${N_SAMPLE:-20}" \
  --out /home/ll5914/Jepa_yifan/JEPA_ARVR/experiments/fastgen_head_heatmaps/encoder_L23_multiclip
