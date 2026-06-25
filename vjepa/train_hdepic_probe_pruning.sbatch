#!/bin/bash
#SBATCH --job-name=hdepic_pruning
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/ll5914/logs/hdepic_pruning_%j.out

export PYTHONUNBUFFERED=1

echo "=== HD-EPIC Probe Fine-tuning  (attention-based token pruning) ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Settings:  keep_ratio=0.50  (4096 → 2048 tokens),  lora_rank=8"
echo "Walltime:  18h | GPU: H200"
echo "Start:     $(date)"
echo ""

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

mkdir -p /scratch/ll5914/logs
mkdir -p /scratch/ll5914/models/vjepa2

cd /home/ll5914/ARVR_Video/vjepa2
python /home/ll5914/ARVR_Video/vjepa/train_hdepic_probe.py

echo ""
echo "Done: $(date)"
