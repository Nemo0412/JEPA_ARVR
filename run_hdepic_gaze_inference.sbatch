#!/bin/bash
#SBATCH --job-name=hdepic_gaze_e
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/ll5914/logs/hdepic_gaze_infer_%j.out

export PYTHONUNBUFFERED=1
# Optional: uncomment for smoke test (first N samples)
# export HDEPIC_GAZE_EVAL_MAX=32
# export HDEPIC_GAZE_BATCH=2

echo "=== HD-EPIC gaze-aware probe eval ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo ""

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

mkdir -p /scratch/ll5914/logs

cd /home/ll5914/ARVR_Video/vjepa2
python /home/ll5914/ARVR_Video/run_hdepic_gaze_inference.py

echo ""
echo "Done: $(date)"
