#!/bin/bash
#SBATCH --job-name=hdepic_sink_ex
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:h100:1 --cpus-per-task=8 --mem=64GB --time=00:20:00
#SBATCH --output=/scratch/ll5914/logs/hdepic_sink_example_%j.out
#SBATCH --error=/scratch/ll5914/logs/hdepic_sink_example_%j.err

set -euo pipefail
export PYTHONPATH="/home/ll5914/Jepa_yifan/JEPA_ARVR:/home/ll5914/ARVR_Video/vjepa2:${PYTHONPATH:-}"
export VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
export TRI_MODAL_FRAME_CACHE="/scratch/ll5914/datasets/HD-EPIC/_clip_frame_cache/p01_f32_at1"
cd /home/ll5914/Jepa_yifan/JEPA_ARVR
/scratch/ll5914/conda_envs/SVD/bin/python scripts/plot_hdepic_sink_corner_example.py
