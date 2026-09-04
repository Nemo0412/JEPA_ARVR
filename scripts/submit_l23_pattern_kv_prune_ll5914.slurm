#!/bin/bash
#SBATCH --job-name=l23_kvprune
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/ll5914/logs/l23_kvprune_%j.out
#SBATCH --error=/scratch/ll5914/logs/l23_kvprune_%j.err

set -euo pipefail
export PYTHONPATH="/home/ll5914/Jepa_yifan/JEPA_ARVR:/home/ll5914/ARVR_Video/vjepa2:${PYTHONPATH:-}"
export TRI_MODAL_FRAME_CACHE="/scratch/ll5914/datasets/HD-EPIC/_clip_frame_cache/p01_f32_at1"
export VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
mkdir -p /scratch/ll5914/logs
cd /home/ll5914/Jepa_yifan/JEPA_ARVR
/scratch/ll5914/conda_envs/SVD/bin/python scripts/test_l23_pattern_kv_prune.py \
  --n-sample "${N_SAMPLE:-12}" \
  --out /home/ll5914/Jepa_yifan/JEPA_ARVR/experiments/fastgen_head_heatmaps/encoder_L23_multiclip/l23_kv_prune_test.json
echo "done $(date)"
