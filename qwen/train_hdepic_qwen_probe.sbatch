#!/bin/bash
#SBATCH --job-name=qwen_probe
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ll5914/logs/qwen_probe_%j.out

# ── Full Qwen2.5-VL-3B (LoRA) — 1× H200 ─────────────────────────────
# Account limit: 1 GPU per user.
# Effective batch = BATCH_SIZE(4) × GRAD_ACCUM(2) = 8
# ─────────────────────────────────────────────────────────────────────

export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/ll5914/hf_cache
export TRANSFORMERS_CACHE=/scratch/ll5914/hf_cache
mkdir -p /scratch/ll5914/hf_cache

echo "=== Qwen2.5-VL-3B HD-EPIC Probe Baseline (1× H200) ==="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Walltime: 24h | 1× H200 | mem: 80 GB"
echo "Start   : $(date)"
echo ""

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

pip install -q "transformers>=4.49.0" 2>/dev/null || true

mkdir -p /scratch/ll5914/logs
mkdir -p /scratch/ll5914/models/vjepa2

# Pick a free port for NCCL rendezvous
export MASTER_PORT=$(python -c "import random,socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")
export MASTER_ADDR=$(hostname)

cd /home/ll5914/ARVR_Video/vjepa2

python /home/ll5914/ARVR_Video/qwen/train_hdepic_qwen_probe.py

echo ""
echo "Done: $(date)"
