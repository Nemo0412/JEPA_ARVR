#!/bin/bash
#SBATCH --job-name=qwen_probe
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ll5914/logs/qwen_probe_%j.out

# ── Qwen2.5-VL-3B visual encoder + AttentivePooler probe ──────────────
# Baseline comparable to train_hdepic_probe.py (V-JEPA ViT-L):
#   - Same HD-EPIC dataset, same 20/2/5 train/val/test split
#   - Same 4s clip + 1s anticipation horizon
#   - Same AttentivePooler (depth-4) + verb/noun/action heads
#   - Only the probe is trained; Qwen visual encoder is frozen
#
# Memory note: Qwen2.5-VL-3B visual tower ~6 GB in BF16;
#   use 64 GB RAM and one H200 (80 GB VRAM).
# ─────────────────────────────────────────────────────────────────────

export PYTHONUNBUFFERED=1

# HuggingFace cache — point to scratch so the 3B model is stored there
export HF_HOME=/scratch/ll5914/hf_cache
export TRANSFORMERS_CACHE=/scratch/ll5914/hf_cache
mkdir -p /scratch/ll5914/hf_cache

echo "=== Qwen2.5-VL-3B HD-EPIC Probe Baseline ==="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Walltime: 24h | GPU: H200 | mem: 64 GB"
echo "Start   : $(date)"
echo ""

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

# Install Qwen2.5-VL dependencies if not already present
pip install -q "qwen-vl-utils>=0.0.8" "transformers>=4.49.0" 2>/dev/null || true

mkdir -p /scratch/ll5914/logs
mkdir -p /scratch/ll5914/models/vjepa2

cd /home/ll5914/ARVR_Video/vjepa2
python /home/ll5914/ARVR_Video/train_hdepic_qwen_probe.py

echo ""
echo "Done: $(date)"
