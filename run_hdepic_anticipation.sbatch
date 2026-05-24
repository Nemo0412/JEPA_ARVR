#!/bin/bash
#SBATCH --job-name=vjepa2_hdepic
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ll5914/logs/vjepa2_hdepic_%j.out

echo "=== V-JEPA 2 HD-EPIC P01 第一个场�?==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo ""

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

mkdir -p /scratch/ll5914/logs

cd /home/ll5914/ARVR_Video/vjepa2
python /home/ll5914/ARVR_Video/run_hdepic_anticipation.py

echo ""
echo "Done: $(date)"
