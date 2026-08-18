#!/bin/bash
# EGTEA data-only port of ll's AGA streaming baseline at upstream f17f7e8.

#SBATCH --job-name=egtea_aga_stream
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ll5914/logs/egtea_aga_stream_%j.out
#SBATCH --error=/scratch/ll5914/logs/egtea_aga_stream_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-/scratch/ll5914/conda_envs/SVD/bin/python}"
EGTEA_ROOT="${EGTEA_ROOT:-/scratch/ll5914/datasets/EGTEA}"
ANN_DIR="${ANN_DIR:-${PROJECT_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1}"
FEAT_DIR="${FEAT_DIR:-${EGTEA_ROOT}/rulstm_features/rgb_ll_tsn_4fps}"
OUT_DIR="${OUT_DIR:-/scratch/ll5914/experiments/egtea_aga_stream}"
TRAINER="${PROJECT_ROOT}/baselines/aga_hdepic/train_stream_aga.py"
MODEL="${PROJECT_ROOT}/baselines/AGA/model.py"

"${PROJECT_ROOT}/scripts/egtea/verify_stream_split.sh"
"${PROJECT_ROOT}/scripts/egtea/verify_rulstm_feature_bundle.sh" "${FEAT_DIR}" "${ANN_DIR}"

[[ "$(sha256sum "${TRAINER}" | awk '{print $1}')" == \
  "8e244e9233a284357a37dedce17adcd947e852bb8140d962b5f45e2cdcc4ebd0" ]]
[[ "$(sha256sum "${MODEL}" | awk '{print $1}')" == \
  "8b5901522dd97c2a15e1b2fe25425bc68edf147883d6f1daccca2f43dab4badf" ]]

mkdir -p "${OUT_DIR}" /scratch/ll5914/logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

exec "${PYTHON}" -u "${TRAINER}" \
  --train-csv "${ANN_DIR}/EGTEA_train_stream_mtp.csv" \
  --val-csv "${ANN_DIR}/EGTEA_val_stream_mtp.csv" \
  --feat-dir "${FEAT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --horizons-sec "2,4,6" \
  --loss-weights "1.0,0.7,0.5" \
  --time-step 1.0 \
  --feat-alpha 0.25 \
  --max-context-sec 10.0 \
  --batch-size 128 \
  --num-workers 4 \
  --epochs 50 \
  --lr 2e-4 \
  --weight-decay 0.01 \
  --amp
