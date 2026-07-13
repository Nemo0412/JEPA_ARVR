#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__ref_egoexo_motion
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/ref_egoexo_motion_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/ref_egoexo_motion_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# REF comparison baseline: frozen V-JEPA2 ViT-g/384 encoder + attentive motion head
# trained from scratch on Ego-Exo4D 3D Human Motion Prediction (EgoAgent window protocol).

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

CONFIG_DIR="${CONFIG_DIR:-${PROJECT_ROOT}/configs/generated}"
CONFIG_PATH="${CONFIG_PATH:-${CONFIG_DIR}/ref_egoexo4d_motion_vitg384.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/ref_egoexo4d_motion_prediction}"
REF_TAG="${REF_TAG:-ref-egoexo-vitg384-motion30-scratch-probe}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vitg-384.pt}"
INDEX_DIR="${INDEX_DIR:-${PROJECT_ROOT}/data/egoexo4d_motion_index}"

EVAL_GPUS="${EVAL_GPUS:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-20}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"
VAL_ONLY="${VAL_ONLY:-0}"
USE_PAST_POSE="${USE_PAST_POSE:-1}"
MOTION_FPS="${MOTION_FPS:-30}"

mkdir -p "${PROJECT_ROOT}/logs" "${CONFIG_DIR}" "${OUTPUT_DIR}"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME CONFIG_DIR CONFIG_PATH OUTPUT_DIR REF_TAG CHECKPOINT INDEX_DIR
export EVAL_GPUS EVAL_BATCH_SIZE EVAL_NUM_EPOCHS EVAL_NUM_WORKERS RESUME_CHECKPOINT VAL_ONLY USE_PAST_POSE MOTION_FPS

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

for split in train val; do
  csv="${INDEX_DIR}/egoexo_motion_${split}_fps${MOTION_FPS}.csv"
  if [[ ! -f "${csv}" ]]; then
    echo "[ref-egoexo] Missing index ${csv}; run scripts/prepare_egoexo4d_motion_index_cpu.slurm first" >&2
    exit 1
  fi
done

python - <<'PY'
import os
from pathlib import Path
import yaml

project_root = Path(os.environ["PROJECT_ROOT"])
config_path = Path(os.environ["CONFIG_PATH"])
index_dir = Path(os.environ["INDEX_DIR"])
motion_fps = int(os.environ["MOTION_FPS"])

with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["tag"] = os.environ["REF_TAG"]
cfg["folder"] = os.environ["OUTPUT_DIR"]
cfg["resume_checkpoint"] = os.environ["RESUME_CHECKPOINT"] == "1"
cfg["val_only"] = os.environ.get("VAL_ONLY", "0") == "1"
cfg["model_kwargs"]["checkpoint"] = os.environ["CHECKPOINT"]

data = cfg["experiment"]["data"]
data["dataset_train"] = str(index_dir / f"egoexo_motion_train_fps{motion_fps}.csv")
data["dataset_val"] = str(index_dir / f"egoexo_motion_val_fps{motion_fps}.csv")

cfg["experiment"]["motion_prediction"]["motion_fps"] = motion_fps
cfg["experiment"]["motion_prediction"]["use_past_pose"] = os.environ.get("USE_PAST_POSE", "1") == "1"

opt = cfg["experiment"]["optimization"]
opt["batch_size"] = int(os.environ["EVAL_BATCH_SIZE"])
opt["num_epochs"] = int(os.environ["EVAL_NUM_EPOCHS"])
data["num_workers"] = int(os.environ["EVAL_NUM_WORKERS"])

with config_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[ref-egoexo] Wrote config: {config_path}")
PY

cd "${VJEPA_ROOT}"
python -m evals.main --fname "${CONFIG_PATH}" --devices cuda:0 --debugmode true
BASH
)

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
