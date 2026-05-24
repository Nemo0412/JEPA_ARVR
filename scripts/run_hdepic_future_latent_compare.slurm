#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_future_latents
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_future_latents_%j.out
#SBATCH --error=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_future_latents_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# Compare latent sources on the same HD-EPIC validation samples:
#   encoder: observed clip encoder tokens only, no predictor call
#   direct_single: one mask span placed at the requested future target chunk
#   direct_dense: one larger mask span covering intermediate chunks through target
#   ar: predictor rollout one valid local chunk at a time
#   oracle: encoder target chunk from the actual future/action-time clip
#
# Outputs one CSV with latent distance-to-oracle and action/verb/noun metrics.

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

ADAPTED_ANNOTATIONS_DIR="${ADAPTED_ANNOTATIONS_DIR:-${PROJECT_ROOT}/data/hdepic_vjepa_annotations}"
ADAPTED_VIDEO_ROOT="${ADAPTED_VIDEO_ROOT:-${PROJECT_ROOT}/data/hdepic_vjepa_videos}"
CONFIG_DIR="${CONFIG_DIR:-${PROJECT_ROOT}/configs/generated/future_latent_compare}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vitg-384.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"
LORA_PRETRAINED_PROBE="${LORA_PRETRAINED_PROBE:-${PROJECT_ROOT}/outputs/ek100_vitg384_probe/action_anticipation_frozen/ek100-vitg16-384/latest.pt}"
LORA_TAG="${LORA_TAG:-hdepic-lora-vitg16-384-cleanp01-refmetrics}"

HORIZONS="${HORIZONS:-1 1.5 2 2.5 3 4 5 6 7 8 9 10 60}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DROP_INCOMPLETE_HISTORY="${DROP_INCOMPLETE_HISTORY:-1}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MAX_ROLLOUT_STEPS="${MAX_ROLLOUT_STEPS:-512}"
RESULT_CSV="${RESULT_CSV:-${OUTPUT_DIR}/future_latent_compare/${LORA_TAG}/future_latent_compare.csv}"
CONFIG_PATH="${CONFIG_PATH:-${CONFIG_DIR}/hdepic_future_latent_compare.yaml}"

mkdir -p "${PROJECT_ROOT}/logs" "${CONFIG_DIR}" "$(dirname "${RESULT_CSV}")"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export ADAPTED_ANNOTATIONS_DIR ADAPTED_VIDEO_ROOT CONFIG_DIR CONFIG_PATH CHECKPOINT OUTPUT_DIR
export LORA_PRETRAINED_PROBE LORA_TAG HORIZONS EVAL_BATCH_SIZE EVAL_NUM_WORKERS MAX_SAMPLES
export DROP_INCOMPLETE_HISTORY LORA_RANK LORA_ALPHA LORA_DROPOUT MAX_ROLLOUT_STEPS RESULT_CSV

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

python - <<'PY'
import os
from pathlib import Path
import yaml

project_root = Path(os.environ["PROJECT_ROOT"])
template_path = project_root / "vjepa2" / "configs" / "eval" / "vitg-384" / "ek100.yaml"
with template_path.open("r", encoding="utf-8") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg["nodes"] = 1
cfg["tasks_per_node"] = 1
cfg["cpus_per_task"] = 12
cfg["tag"] = os.environ["LORA_TAG"]
cfg["eval_name"] = "app.hdepic_lora_action_anticipation"
cfg["folder"] = os.environ["OUTPUT_DIR"]
cfg["resume_checkpoint"] = True
cfg["val_only"] = True

data = cfg["experiment"]["data"]
data["dataset"] = "EK100"
data["file_format"] = 1
data["base_path"] = os.environ["ADAPTED_VIDEO_ROOT"]
data["dataset_train"] = str(Path(os.environ["ADAPTED_ANNOTATIONS_DIR"]) / "HD_EPIC_train_vjepa.csv")
data["dataset_val"] = str(Path(os.environ["ADAPTED_ANNOTATIONS_DIR"]) / "HD_EPIC_val_vjepa.csv")
data["anticipation_time_sec"] = [1.0, 1.0]
data["num_workers"] = int(os.environ["EVAL_NUM_WORKERS"])
data["val_anticipation_point"] = [0.0, 0.0]

cfg["experiment"]["optimization"]["batch_size"] = int(os.environ["EVAL_BATCH_SIZE"])
cfg["experiment"]["optimization"]["num_epochs"] = 1
cfg["model_kwargs"]["checkpoint"] = os.environ["CHECKPOINT"]
cfg["model_kwargs"]["module_name"] = "app.hdepic_lora_action_anticipation.modelcustom.vit_encoder_predictor_rollout"
cfg["model_kwargs"]["wrapper_kwargs"]["num_steps"] = 1
cfg["model_kwargs"]["wrapper_kwargs"]["num_output_frames"] = 2
cfg["model_kwargs"]["wrapper_kwargs"]["max_rollout_steps"] = int(os.environ["MAX_ROLLOUT_STEPS"])

cfg["experiment"]["lora"] = {
    "enabled": True,
    "rank": int(os.environ["LORA_RANK"]),
    "alpha": float(os.environ["LORA_ALPHA"]),
    "dropout": float(os.environ["LORA_DROPOUT"]),
    "train_heads": True,
    "pretrained_probe": os.environ["LORA_PRETRAINED_PROBE"],
    "align_reference_metrics": True,
}

out = Path(os.environ["CONFIG_PATH"])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[future-latents] wrote {out}")
PY

python -m app.hdepic_lora_action_anticipation.future_latent_compare \
    --config "${CONFIG_PATH}" \
    --horizons "${HORIZONS}" \
    --out "${RESULT_CSV}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${EVAL_NUM_WORKERS}" \
    --max-samples "${MAX_SAMPLES}" \
    $(if [[ "${DROP_INCOMPLETE_HISTORY}" == "1" ]]; then echo "--drop-incomplete-history"; fi)
BASH
)

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
