#!/bin/bash
#SBATCH --job-name=hdepic_train
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5914/logs/hdepic_train_%j.out

export PYTHONUNBUFFERED=1

echo "=== HD-EPIC Probe Fine-tuning ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Walltime: 12h | GPU: H200 | ckpt: hdepic-vitl-probe-last.pt + hdepic-vitl-probe-best.pt"
echo "Start: $(date)"
echo ""

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

mkdir -p /scratch/ll5914/logs
mkdir -p /scratch/ll5914/models/vjepa2

cd /home/ll5914/ARVR_Video/vjepa2
python /home/ll5914/ARVR_Video/vjepa/train_hdepic_probe.py

echo ""
echo "Done: $(date)"
