#!/bin/bash
#SBATCH --job-name=p01_stream_mtp
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=384GB
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ll5914/logs/p01_stream_mtp_%j.out
#SBATCH --error=/scratch/ll5914/logs/p01_stream_mtp_%j.err

set -euo pipefail

# Streaming MTP (communicating MLP @ +2/+4/+6s):
#   - temporal half-split per video (1st half train / 2nd half val)
#   - growing context 4→6→8→10s from half origin, then slide 10s; tick every 2s
#   - prune-before-predictor (keep≤4096) when long contexts overflow predictor budget
# Separate from fixed-clip anticipative eval; does not change that path.

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
export VJEPA_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

VIDEO_ROOT="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"

VIDEO_BEST_DIR="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep"
ENCODER_LORA="${VIDEO_BEST_DIR}/encoder_lora_best.pt"
PREDICTOR_LORA="${VIDEO_BEST_DIR}/predictor_lora_best.pt"

OUT_DIR="/scratch/ll5914/experiments/p01_stream_mtp_2_4_6"
mkdir -p /scratch/ll5914/logs "${OUT_DIR}" "${ANN_DIR}"

VAL_ONLY="${VAL_ONLY:-0}"

echo "========================================================"
echo "P01 streaming MTP 2/4/6 (half-split + grow context + prune)"
echo "Job=${SLURM_JOB_ID:-manual} VAL_ONLY=${VAL_ONLY}"
echo "ANN=${ANN_DIR}"
echo "OUT=${OUT_DIR}"
echo "Start: $(date)"
echo "========================================================"

if [[ ! -f "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv" ]]; then
  echo "Building stream half-split index..."
  "${PYTHON}" "${PROJECT_ROOT}/scripts/make_hdepic_stream_half_split.py" \
    --out "${ANN_DIR}"
fi

ARGS=(
  --train-csv "${ANN_DIR}/HD_EPIC_train_stream_mtp.csv"
  --val-csv "${ANN_DIR}/HD_EPIC_val_stream_mtp.csv"
  --video-root "${VIDEO_ROOT}"
  --checkpoint "${CHECKPOINT}"
  --encoder-lora "${ENCODER_LORA}"
  --predictor-lora "${PREDICTOR_LORA}"
  --out-dir "${OUT_DIR}"
  --horizons-sec "2,4,6"
  --loss-weights "1.0,0.7,0.5"
  --primary-horizon-sec 2.0
  --anticipation-sec 2.0
  --max-frames 80
  --fps 8
  --img-size 256
  --keep-count 4096
  --batch-size 2
  --num-workers 6
  --epochs 8
  --lr 2e-5
)

if [[ "${VAL_ONLY}" == "1" ]]; then
  ARGS+=(--val-only)
fi

cd "${PROJECT_ROOT}"
"${PYTHON}" -m app.hdepic_lora_action_anticipation.train_stream_mtp "${ARGS[@]}"

echo "Done: $(date) VAL_ONLY=${VAL_ONLY}"
