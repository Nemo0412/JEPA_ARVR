#!/bin/bash
#SBATCH --job-name=JEPA_ARVR__enc_exit_entropy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192GB
#SBATCH --time=00:45:00
#SBATCH --output=logs/enc_exit_entropy_%j.out
#SBATCH --error=logs/enc_exit_entropy_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEPTHS="${DEPTHS:-6,12,18,24}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual}}"
CONFIG="${CONFIG:?Set CONFIG to the parent checkpoint training YAML}"
CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to the directory containing best.pt and LoRA sidecars}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/encoder_early_exit_entropy/video_${RUN_ID}}"
EXPECTED_SPLIT_DIR="${EXPECTED_SPLIT_DIR:?Set EXPECTED_SPLIT_DIR to the checkpoint-bound frozen split directory}"
SPLIT_LABEL="${SPLIT_LABEL:-checkpoint_bound_split}"

export PROJECT_ROOT SIF OVERLAY ENV_NAME MAX_SAMPLES BATCH_SIZE NUM_WORKERS DEPTHS RUN_ID CONFIG CKPT_DIR OUT_DIR EXPECTED_SPLIT_DIR SPLIT_LABEL

singularity exec --nv --overlay "${OVERLAY}:ro" "${SIF}" /bin/bash -lc 'bash -s' <<'EOF'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate "/ext3/envs/${ENV_NAME}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
python scripts/analyze_predictor_early_exit_entropy.py \
  --exit-module encoder \
  --config "${CONFIG}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --expected-split-dir "${EXPECTED_SPLIT_DIR}" \
  --split-label "${SPLIT_LABEL}" \
  --depths "${DEPTHS}" \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"
EOF
