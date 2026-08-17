#!/bin/bash
# EGTEA data-port of ll's optimized shallow-wide RU-LSTM Large v2.

#SBATCH --job-name=egtea_rulstm_large_v2
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ll5914/logs/egtea_rulstm_large_v2_%j.out
#SBATCH --error=/scratch/ll5914/logs/egtea_rulstm_large_v2_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-/scratch/ll5914/conda_envs/SVD/bin/python}"
EGTEA_ROOT="${EGTEA_ROOT:-/scratch/ll5914/datasets/EGTEA}"
ANN_DIR="${ANN_DIR:-${PROJECT_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1}"
FEAT_DIR="${FEAT_DIR:-${EGTEA_ROOT}/rulstm_features/rgb_ll_tsn_4fps}"
INIT_FROM="${INIT_FROM:?Set INIT_FROM to the EGTEA small RU-LSTM best checkpoint}"
OUT_DIR="${OUT_DIR:-/scratch/ll5914/experiments/egtea_rulstm_large_v2}"
TRAINER="${PROJECT_ROOT}/baselines/rulstm_hdepic/train_stream_rulstm.py"

"${PROJECT_ROOT}/scripts/egtea/verify_stream_split.sh"
[[ -f "${INIT_FROM}" ]] || { echo "ERROR: missing ${INIT_FROM}" >&2; exit 2; }
"${PROJECT_ROOT}/scripts/egtea/verify_rulstm_feature_bundle.sh" "${FEAT_DIR}" "${ANN_DIR}"
mkdir -p "${OUT_DIR}" /scratch/ll5914/logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

exec "${PYTHON}" "${TRAINER}" \
  --train-csv "${ANN_DIR}/EGTEA_train_stream_mtp.csv" \
  --val-csv "${ANN_DIR}/EGTEA_val_stream_mtp.csv" \
  --feat-dir "${FEAT_DIR}" --out-dir "${OUT_DIR}" \
  --horizons-sec "2,4,6" --loss-weights "1.0,0.7,0.5" \
  --alpha 0.25 --max-context-sec 10.0 \
  --hidden 4096 --depth 1 --input-proj --mlp-head --trunk-layers 0 \
  --dropout 0.5 --optimizer adamw --lr 1e-3 --weight-decay 0.05 \
  --lr-schedule cosine --warmup-epochs 3 --label-smoothing 0.1 \
  --feat-noise 0.05 --grad-clip 1.0 --amp --early-stop-patience 12 \
  --init-from "${INIT_FROM}" --batch-size 8 --num-workers 4 --epochs 40
