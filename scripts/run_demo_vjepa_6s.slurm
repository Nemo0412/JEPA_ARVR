#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__demo_vjepa_6s
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:20:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/demo_vjepa_6s_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/demo_vjepa_6s_%j.err

set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP//path/to/VJEPA2-EXP}"
export SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"
export ENV_NAME="${ENV_NAME:-EgoLifeExp}"
export ROW="${ROW:-0}"
export NUM_FRAMES="${NUM_FRAMES:-48}"

export HF_HOME="${HF_HOME:-/path/to/.huggingface}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH=\"${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:\${PYTHONPATH:-}\"
cd \"${PROJECT_ROOT}\"
python scripts/demo_vjepa_6s.py --row \${ROW} --num-frames \${NUM_FRAMES}
"
