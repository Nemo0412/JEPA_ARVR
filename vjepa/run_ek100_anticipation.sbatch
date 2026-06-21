#!/bin/bash
#SBATCH -J vjepa2_ek100
#SBATCH -c 4
#SBATCH -t 00:20:00
#SBATCH --mem=32G
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH -o /scratch/ll5914/logs/vjepa2_ek100_%j.out
#SBATCH -e /scratch/ll5914/logs/vjepa2_ek100_%j.err

echo "Job started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

cd /home/ll5914/ARVR_Video/vjepa2
python ../run_ek100_anticipation.py

echo "Job finished: $(date)"
