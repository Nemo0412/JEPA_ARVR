#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_probe_val
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_probe_val_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_probe_val_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Val-only HD-EPIC smoke test using the official EK100 ViT-g/384 attentive probe.
# Requires:
#   bash scripts/download_ek100_vitg384_inference_ckpts.sh
#
# Default horizons are 1s and 10s. 60s is not run by default because the
# released predictor uses a finite temporal position range and is expected to
# exceed it at one minute.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

ADAPTED_ANNOTATIONS_DIR="${ADAPTED_ANNOTATIONS_DIR:-${PROJECT_ROOT}/data/hdepic_vjepa_annotations}"
ADAPTED_VIDEO_ROOT="${ADAPTED_VIDEO_ROOT:-${PROJECT_ROOT}/data/hdepic_vjepa_videos}"
CONFIG_DIR="${CONFIG_DIR:-${PROJECT_ROOT}/configs/generated}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vitg-384.pt}"
PROBE_ROOT="${PROBE_ROOT:-${PROJECT_ROOT}/outputs/ek100_vitg384_probe}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/outputs/hdepic_ek100_probe_valonly}"
HORIZONS="${HORIZONS:-1,10}"
RUN_LONG_HORIZON="${RUN_LONG_HORIZON:-0}"
LONG_HORIZON="${LONG_HORIZON:-60}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"

mkdir -p "${PROJECT_ROOT}/logs" "${CONFIG_DIR}" "${RESULT_ROOT}"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export ADAPTED_ANNOTATIONS_DIR ADAPTED_VIDEO_ROOT CONFIG_DIR ENCODER_CHECKPOINT PROBE_ROOT RESULT_ROOT
export HORIZONS RUN_LONG_HORIZON LONG_HORIZON EVAL_BATCH_SIZE EVAL_NUM_WORKERS

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}

cd "${PROJECT_ROOT}"

python - <<'PY'
import os
from pathlib import Path
import yaml

project_root = Path(os.environ["PROJECT_ROOT"])
template_path = project_root / "vjepa2" / "configs" / "inference" / "vitg-384" / "ek100.yaml"
config_dir = Path(os.environ["CONFIG_DIR"])
result_root = Path(os.environ["RESULT_ROOT"])

horizons = [h.strip() for h in os.environ["HORIZONS"].split(",") if h.strip()]
if os.environ.get("RUN_LONG_HORIZON") == "1":
    horizons.append(os.environ.get("LONG_HORIZON", "60"))

for horizon in horizons:
    h = float(horizon)
    h_tag = str(int(h)) if h.is_integer() else str(h).replace(".", "p")
    with template_path.open("r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg["nodes"] = 1
    cfg["tasks_per_node"] = 1
    cfg["cpus_per_task"] = 12
    cfg["tag"] = "ek100-vitg16-384"
    cfg["folder"] = os.environ["PROBE_ROOT"]
    cfg["resume_checkpoint"] = True
    cfg["val_only"] = True

    data = cfg["experiment"]["data"]
    data["dataset"] = "EK100"
    data["file_format"] = 1
    data["base_path"] = os.environ["ADAPTED_VIDEO_ROOT"]
    data["dataset_train"] = str(Path(os.environ["ADAPTED_ANNOTATIONS_DIR"]) / "HD_EPIC_train_vjepa.csv")
    data["dataset_val"] = str(Path(os.environ["ADAPTED_ANNOTATIONS_DIR"]) / "HD_EPIC_val_vjepa.csv")
    data["anticipation_time_sec"] = [h, h]
    data["num_workers"] = int(os.environ["EVAL_NUM_WORKERS"])

    cfg["experiment"]["optimization"]["batch_size"] = int(os.environ["EVAL_BATCH_SIZE"])
    cfg["model_kwargs"]["checkpoint"] = os.environ["ENCODER_CHECKPOINT"]

    out_path = config_dir / f"hdepic_ek100_probe_valonly_{h_tag}s.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(out_path)
PY

cd "${VJEPA_ROOT}"
IFS=',' read -ra horizon_array <<< "${HORIZONS}"
if [[ "${RUN_LONG_HORIZON}" == "1" ]]; then
    horizon_array+=("${LONG_HORIZON}")
fi

for horizon in "${horizon_array[@]}"; do
    h_tag="${horizon%.*}"
    config_path="${CONFIG_DIR}/hdepic_ek100_probe_valonly_${h_tag}s.yaml"
    echo "[val_only] Running horizon ${horizon}s with ${config_path}"
    python -m evals.main \
        --fname "${config_path}" \
        --devices cuda:0 \
        --debugmode true
done
BASH
)

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
