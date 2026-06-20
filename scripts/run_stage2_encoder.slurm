#!/bin/bash
#SBATCH --job-name=p01_stage2_enc
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/ll5914/logs/stage2_encoder_%j.out
#SBATCH --error=/scratch/ll5914/logs/stage2_encoder_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Encoder + probe LoRA fine-tuning with 19/2/5 split.
#
# - eval_name: app.hdepic_lora_encoder_action_anticipation
# - Probe init: Stage 1 checkpoint (learned all P01 vocab)
# - Encoder LoRA rank=4 on attention layers (qkv/proj)
# - Probe LoRA rank=8 on pooler (continues from Stage 1)
# - Train on 19 videos, validate on 2, hold 5 for final test
# - After training: automatic val_only run on 5 test videos
#
# Run after Stage 1 completes:
#   STEP1=$(sbatch scripts/run_stage1_probe.slurm | awk '{print $NF}')
#   sbatch --dependency=afterok:$STEP1 scripts/run_stage2_encoder.slurm
# ─────────────────────────────────────────────────────────────────────────────

VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
APP_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
ANNOTATIONS_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations"
VIDEO_ROOT="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
ENCODER_CKPT="/scratch/ll5914/models/vjepa2/vitl.pt"
STAGE1_PROBE_CKPT="/scratch/ll5914/experiments/hdepic_two_stage/action_anticipation_frozen/p01-stage1-probe-vitl16-256/latest.pt"
OUTPUT_DIR="/scratch/ll5914/experiments/hdepic_two_stage"
LORA_TAG="p01-stage2-enc-vitl16-256"
CONFIG_TRAIN="/scratch/ll5914/logs/stage2_encoder_train_config.yaml"
CONFIG_TEST="/scratch/ll5914/logs/stage2_encoder_test_config.yaml"

# Probe / pooler LoRA
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# Encoder LoRA
ENC_LORA_RANK=4
ENC_LORA_ALPHA=8
ENC_LORA_DROPOUT=0.0
ENC_LORA_TARGET="attn"   # only attention linear layers
ENC_LORA_LR=2e-5
ENC_LORA_WD=0.0

BATCH_SIZE=1
NUM_EPOCHS=3
NUM_WORKERS=4

mkdir -p /scratch/ll5914/logs "${OUTPUT_DIR}"

echo "=== Stage 2: Encoder + Probe LoRA fine-tuning ==="
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo ""

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

export PYTHONPATH="${APP_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Sanity checks
[[ -f "${STAGE1_PROBE_CKPT}" ]] || {
    echo "ERROR: Stage 1 probe checkpoint not found: ${STAGE1_PROBE_CKPT}"
    echo "Run stage 1 first, or set STAGE1_PROBE_CKPT."
    exit 1
}
for f in "${ANNOTATIONS_DIR}/stage2/HD_EPIC_train_vjepa.csv" \
          "${ANNOTATIONS_DIR}/stage2/HD_EPIC_val_vjepa.csv"   \
          "${ANNOTATIONS_DIR}/stage2/HD_EPIC_test_vjepa.csv"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done

# ── Generate training config ──────────────────────────────────────────────────
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
cfg["eval_name"] = "app.hdepic_lora_encoder_action_anticipation"
cfg["folder"] = "${OUTPUT_DIR}"
cfg["resume_checkpoint"] = True
cfg["val_only"] = False

d = cfg["experiment"]["data"]
d["dataset"] = "EK100"
d["file_format"] = 1
d["base_path"] = "${VIDEO_ROOT}"
d["dataset_train"] = "${ANNOTATIONS_DIR}/stage2/HD_EPIC_train_vjepa.csv"
d["dataset_val"]   = "${ANNOTATIONS_DIR}/stage2/HD_EPIC_val_vjepa.csv"
d["num_workers"] = ${NUM_WORKERS}

opt = cfg["experiment"]["optimization"]
opt["batch_size"] = ${BATCH_SIZE}
opt["num_epochs"] = ${NUM_EPOCHS}
opt["multihead_kwargs"] = [opt["multihead_kwargs"][0]]

cfg["model_kwargs"]["checkpoint"] = "${ENCODER_CKPT}"

cfg["experiment"]["lora"] = {
    "enabled": True,
    "rank": ${LORA_RANK},
    "alpha": ${LORA_ALPHA},
    "dropout": ${LORA_DROPOUT},
    "train_heads": True,
    "pretrained_probe": "${STAGE1_PROBE_CKPT}",
    "align_reference_metrics": True,
}

enc_target = ["${ENC_LORA_TARGET}"] if "${ENC_LORA_TARGET}" else None
cfg["experiment"]["encoder_lora"] = {
    "enabled": True,
    "rank": ${ENC_LORA_RANK},
    "alpha": ${ENC_LORA_ALPHA},
    "dropout": ${ENC_LORA_DROPOUT},
    "target_modules": enc_target,
    "lr": ${ENC_LORA_LR},
    "weight_decay": ${ENC_LORA_WD},
}

out = Path("${CONFIG_TRAIN}")
with out.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"Train config written: {out}")
PY

# ── Training pass ─────────────────────────────────────────────────────────────
echo ""
echo "[stage2] Training on 19 videos..."
cd "${VJEPA_ROOT}"
python -m evals.main \
    --fname "${CONFIG_TRAIN}" \
    --devices cuda:0 \
    --debugmode true

# ── Test evaluation on 5 held-out videos ─────────────────────────────────────
python - <<PY
import yaml
from pathlib import Path

with Path("${CONFIG_TRAIN}").open() as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg["val_only"] = True
cfg["resume_checkpoint"] = True
cfg["experiment"]["data"]["dataset_val"] = \
    "${ANNOTATIONS_DIR}/stage2/HD_EPIC_test_vjepa.csv"

out = Path("${CONFIG_TEST}")
with out.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"Test config written: {out}")
PY

echo ""
echo "[stage2] Test evaluation on 5 held-out videos..."
cd "${VJEPA_ROOT}"
python -m evals.main \
    --fname "${CONFIG_TEST}" \
    --devices cuda:0 \
    --debugmode true

echo ""
echo "=== Stage 2 Done: $(date) ==="
echo "Checkpoint: ${OUTPUT_DIR}/action_anticipation_frozen/${LORA_TAG}/latest.pt"
