#!/bin/bash
#SBATCH -J vjepa2_inference
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH --mem=32G
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH -o /scratch/ll5914/logs/vjepa2_inference_%j.out
#SBATCH -e /scratch/ll5914/logs/vjepa2_inference_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

cd /home/ll5914/ARVR_Video/vjepa2
python ../run_inference.py

echo "Job finished: $(date)"
