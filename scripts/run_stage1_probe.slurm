#!/bin/bash
#SBATCH --job-name=p01_stage1_probe
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/ll5914/logs/stage1_probe_%j.out
#SBATCH --error=/scratch/ll5914/logs/stage1_probe_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Probe vocab learning on ALL 26 annotated P01 videos.
#
# Goal: train the AttentiveClassifier (probe/pooler) to recognise every
# P01 action/verb/noun class.  Encoder stays completely frozen.
#
# - eval_name: app.hdepic_lora_action_anticipation  (probe-only LoRA)
# - Encoder:   ViT-L, checkpoint vitl.pt, resolution 256
# - Probe init: EK100 ViT-L/256 pooler weights (ek100-vitl-256.pt)
# - LoRA rank=8 on probe/pooler; classification heads trained freely
# - train: 26 videos (all P01, full vocab), val: 2 videos (monitoring)
#
# Output checkpoint: used as STAGE1_PROBE_CKPT in Stage 2.
# ─────────────────────────────────────────────────────────────────────────────

VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
APP_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
ANNOTATIONS_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations"
VIDEO_ROOT="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
ENCODER_CKPT="/scratch/ll5914/models/vjepa2/vitl.pt"
EK100_PROBE_CKPT="/scratch/ll5914/models/vjepa2/ek100-vitl-256.pt"
OUTPUT_DIR="/scratch/ll5914/experiments/hdepic_two_stage"
LORA_TAG="p01-stage1-probe-vitl16-256"
CONFIG_PATH="/scratch/ll5914/logs/stage1_probe_config.yaml"

LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
BATCH_SIZE=2
NUM_EPOCHS=5
NUM_WORKERS=4

mkdir -p /scratch/ll5914/logs "${OUTPUT_DIR}"

echo "=== Stage 1: Probe vocab learning ==="
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo ""

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

export PYTHONPATH="${APP_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Sanity check
for f in "${ANNOTATIONS_DIR}/stage1/HD_EPIC_train_vjepa.csv" \
          "${ANNOTATIONS_DIR}/stage1/HD_EPIC_val_vjepa.csv"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done

# Generate config
python - <<PY
import yaml
from pathlib import Path

template = Path("${VJEPA_ROOT}/configs/eval/vitl/ek100.yaml")
with template.open() as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg["nodes"] = 1
cfg["tasks_per_node"] = 1
cfg["cpus_per_task"] = 12
cfg["tag"] = "${LORA_TAG}"
cfg["eval_name"] = "app.hdepic_lora_action_anticipation"
cfg["folder"] = "${OUTPUT_DIR}"
cfg["resume_checkpoint"] = True
cfg["val_only"] = False

d = cfg["experiment"]["data"]
d["dataset"] = "EK100"
d["file_format"] = 1
d["base_path"] = "${VIDEO_ROOT}"
d["dataset_train"] = "${ANNOTATIONS_DIR}/stage1/HD_EPIC_train_vjepa.csv"
d["dataset_val"]   = "${ANNOTATIONS_DIR}/stage1/HD_EPIC_val_vjepa.csv"
d["num_workers"] = ${NUM_WORKERS}

opt = cfg["experiment"]["optimization"]
opt["batch_size"] = ${BATCH_SIZE}
opt["num_epochs"] = ${NUM_EPOCHS}
# Keep a single multihead_kwargs entry (one classifier)
opt["multihead_kwargs"] = [opt["multihead_kwargs"][0]]

cfg["model_kwargs"]["checkpoint"] = "${ENCODER_CKPT}"

cfg["experiment"]["lora"] = {
    "enabled": True,
    "rank": ${LORA_RANK},
    "alpha": ${LORA_ALPHA},
    "dropout": ${LORA_DROPOUT},
    "train_heads": True,
    "pretrained_probe": "${EK100_PROBE_CKPT}",
    "align_reference_metrics": True,
}

out = Path("${CONFIG_PATH}")
with out.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"Config written: {out}")
PY

echo ""
echo "[stage1] Training probe on 26 P01 videos..."
cd "${VJEPA_ROOT}"
python -m evals.main \
    --fname "${CONFIG_PATH}" \
    --devices cuda:0 \
    --debugmode true

echo ""
echo "=== Stage 1 Done: $(date) ==="
echo "Probe checkpoint:"
ls -lh "${OUTPUT_DIR}/action_anticipation_frozen/${LORA_TAG}/latest.pt" 2>/dev/null || \
    find "${OUTPUT_DIR}" -name "latest.pt" | head -3
