#!/bin/bash
#SBATCH --job-name=hdepic_lora_enc
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_lora_enc_%j.out
#SBATCH --error=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_lora_enc_%j.err

set -euo pipefail

# LoRA fine-tuning for BOTH the encoder AND the probe/pooler.
# eval_name=app.hdepic_lora_encoder_action_anticipation
#
# Compared with run_hdepic_lora_probe.slurm:
#   - Applies LoRA adapters to the encoder attention layers in addition to the pooler.
#   - EnableGradWrapper routes gradients back through the encoder during training.
#   - encoder_lora section is added to the generated YAML config.
#   - Checkpoints include an "encoder_lora" key with the encoder LoRA state dict.

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

ADAPTED_ANNOTATIONS_DIR="${ADAPTED_ANNOTATIONS_DIR:-${PROJECT_ROOT}/data/hdepic_vjepa_annotations}"
ADAPTED_VIDEO_ROOT="${ADAPTED_VIDEO_ROOT:-${PROJECT_ROOT}/data/hdepic_vjepa_videos}"
CONFIG_DIR="${CONFIG_DIR:-${PROJECT_ROOT}/configs/generated}"
CONFIG_PATH="${CONFIG_PATH:-${CONFIG_DIR}/hdepic_lora_encoder_vitg384.yaml}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vitg-384.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_encoder_action_anticipation}"
LORA_PRETRAINED_PROBE="${LORA_PRETRAINED_PROBE:-${PROJECT_ROOT}/outputs/ek100_vitg384_probe/action_anticipation_frozen/ek100-vitg16-384/latest.pt}"
LORA_TAG="${LORA_TAG:-hdepic-lora-enc-vitg16-384}"

# Probe / pooler LoRA hyper-parameters (same as probe-only run)
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Encoder LoRA hyper-parameters
ENC_LORA_RANK="${ENC_LORA_RANK:-4}"
ENC_LORA_ALPHA="${ENC_LORA_ALPHA:-8}"
ENC_LORA_DROPOUT="${ENC_LORA_DROPOUT:-0.0}"
# Comma-separated substrings to match encoder layer names; empty string = all Linear.
# Default "attn" targets attention qkv/proj blocks only.
ENC_LORA_TARGET="${ENC_LORA_TARGET:-attn}"
ENC_LORA_LR="${ENC_LORA_LR:-2e-5}"
ENC_LORA_WD="${ENC_LORA_WD:-0.0}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-3}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"

mkdir -p "${PROJECT_ROOT}/logs" "${CONFIG_DIR}" "${OUTPUT_DIR}"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export ADAPTED_ANNOTATIONS_DIR ADAPTED_VIDEO_ROOT CONFIG_DIR CONFIG_PATH CHECKPOINT OUTPUT_DIR LORA_PRETRAINED_PROBE
export LORA_TAG LORA_RANK LORA_ALPHA LORA_DROPOUT
export ENC_LORA_RANK ENC_LORA_ALPHA ENC_LORA_DROPOUT ENC_LORA_TARGET ENC_LORA_LR ENC_LORA_WD
export EVAL_BATCH_SIZE EVAL_NUM_EPOCHS EVAL_NUM_WORKERS

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

if [[ ! -f "${PROJECT_ROOT}/app/hdepic_lora_encoder_action_anticipation/eval.py" ]]; then
    echo "[lora-enc] Missing ${PROJECT_ROOT}/app/hdepic_lora_encoder_action_anticipation/eval.py" >&2
    echo "[lora-enc] Sync the project-level app/ directory to HPC before running." >&2
    exit 1
fi

python - <<'PY'
import importlib, sys
print("[lora-enc] sys.path head:", sys.path[:5])
importlib.import_module("app.hdepic_lora_encoder_action_anticipation.eval")
print("[lora-enc] app.hdepic_lora_encoder_action_anticipation import OK")
PY

python - <<'PY'
import os
from pathlib import Path
import yaml

project_root = Path(os.environ["PROJECT_ROOT"])
template_path = project_root / "vjepa2" / "configs" / "eval" / "vitg-384" / "ek100.yaml"
config_path = Path(os.environ["CONFIG_PATH"])

with template_path.open("r", encoding="utf-8") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg["nodes"] = 1
cfg["tasks_per_node"] = 1
cfg["cpus_per_task"] = 12
cfg["tag"] = os.environ["LORA_TAG"]
cfg["eval_name"] = "app.hdepic_lora_encoder_action_anticipation"
cfg["folder"] = os.environ["OUTPUT_DIR"]
cfg["resume_checkpoint"] = False
cfg["val_only"] = False

data = cfg["experiment"]["data"]
data["dataset"] = "EK100"
data["file_format"] = 1
data["base_path"] = os.environ["ADAPTED_VIDEO_ROOT"]
data["dataset_train"] = str(Path(os.environ["ADAPTED_ANNOTATIONS_DIR"]) / "HD_EPIC_train_vjepa.csv")
data["dataset_val"] = str(Path(os.environ["ADAPTED_ANNOTATIONS_DIR"]) / "HD_EPIC_val_vjepa.csv")
data["num_workers"] = int(os.environ["EVAL_NUM_WORKERS"])

cfg["experiment"]["optimization"]["batch_size"] = int(os.environ["EVAL_BATCH_SIZE"])
cfg["experiment"]["optimization"]["num_epochs"] = int(os.environ["EVAL_NUM_EPOCHS"])
cfg["model_kwargs"]["checkpoint"] = os.environ["CHECKPOINT"]

# Probe / pooler LoRA
cfg["experiment"]["lora"] = {
    "enabled": True,
    "rank": int(os.environ["LORA_RANK"]),
    "alpha": float(os.environ["LORA_ALPHA"]),
    "dropout": float(os.environ["LORA_DROPOUT"]),
    "train_heads": True,
    "pretrained_probe": os.environ["LORA_PRETRAINED_PROBE"],
    "align_reference_metrics": True,
}

# Encoder LoRA
enc_target_raw = os.environ["ENC_LORA_TARGET"].strip()
enc_target = [s.strip() for s in enc_target_raw.split(",") if s.strip()] if enc_target_raw else []
cfg["experiment"]["encoder_lora"] = {
    "enabled": True,
    "rank": int(os.environ["ENC_LORA_RANK"]),
    "alpha": float(os.environ["ENC_LORA_ALPHA"]),
    "dropout": float(os.environ["ENC_LORA_DROPOUT"]),
    "target_modules": enc_target if enc_target else None,
    "lr": float(os.environ["ENC_LORA_LR"]),
    "weight_decay": float(os.environ["ENC_LORA_WD"]),
}

config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"[lora-enc] Wrote config: {config_path}")
PY

cd "${VJEPA_ROOT}"
python -m evals.main \
    --fname "${CONFIG_PATH}" \
    --devices cuda:0 \
    --debugmode true
BASH
)

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
