#!/bin/bash
#SBATCH --job-name=hdepic_pose_ev
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/ll5914/logs/hdepic_pose_eval_%j.out

export PYTHONUNBUFFERED=1
export HDEPIC_ANTICIPATION_SEC=10.0
export HDEPIC_AR_STEPS=10
export HDEPIC_POSE_LAYERS=2
# export HDEPIC_EVAL_MAX=32   # smoke test

echo "=== HD-EPIC Pose-aware Anticipation Inference (${HDEPIC_ANTICIPATION_SEC}s) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

mkdir -p /scratch/ll5914/logs

cd /home/ll5914/ARVR_Video/vjepa2
python /home/ll5914/ARVR_Video/vjepa/run_hdepic_anticipation10s_pose.py

echo "Done: $(date)"
