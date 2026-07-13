#!/bin/bash
set -euo pipefail

# RGB-only JEPA_ARVR-style baseline: no input adapter, direct loss.backward through
# encoder LoRA (train_one_epoch_encoder_lora). Grad snapshots every 10 iters.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
TAG="${LORA_TAG:-hdepic-baseline-enclora-graddiag-smoke-bf16-i150}"

export_csv="ALL,PROJECT_ROOT=${PROJECT_ROOT},GAZE_MODE=none,ENCODER_LORA_ENABLED=1,ENCODER_LORA_RANK=8,ENCODER_LORA_ALPHA=16.0,ENCODER_LORA_LAST_N_BLOCKS=0,ENCODER_LORA_LR_MULT=0.5,ENCODER_LORA_TARGET_SUFFIXES=attn.qkv,attn.proj,ENCODER_LORA_ACTIVATION_CHECKPOINTING=1,LORA_PROBE_TRAIN_MODE=full,EVAL_SINGLE_PROBE=1,EVAL_LR=0.0001,EVAL_BATCH_SIZE=4,EVAL_GRAD_CLIP=1.0,EVAL_WARMUP_EPOCHS=2,EVAL_MAX_TRAIN_ITERS=150,EVAL_NUM_EPOCHS=1,EVAL_NUM_WORKERS=2,EVAL_GRAD_DIAG_INTERVAL=10,EVAL_USE_BFLOAT16=1,LORA_PRETRAINED_PROBE=,LORA_TAG=${TAG},CONFIG_PATH=${PROJECT_ROOT}/configs/generated/hdepic_baseline_enclora_graddiag_smoke.yaml,RESUME_CHECKPOINT=0,OUTPUT_DIR=${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation"

echo "[submit-baseline-smoke] tag=${TAG} GAZE_MODE=none direct_backward=encoder_lora_loop"
sbatch \
  --partition=h100_tandon \
  --account=your_slurm_account \
  --gres=gpu:h100:1 \
  --cpus-per-task=12 \
  --mem=768GB \
  --time=12:00:00 \
  --job-name=VJEPA2-EXP__baseline_enclora_smoke \
  --output="${PROJECT_ROOT}/logs/hdepic_baseline_enclora_smoke_%j.out" \
  --error="${PROJECT_ROOT}/logs/hdepic_baseline_enclora_smoke_%j.err" \
  --export="${export_csv}" \
  "${RUN_SCRIPT}"
