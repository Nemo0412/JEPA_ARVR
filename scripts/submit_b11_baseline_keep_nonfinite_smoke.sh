#!/bin/bash
set -euo pipefail

# RGB baseline: keep non-finite encoder-LoRA grads (PhD experiment).
# Compare loss/acc convergence vs discard path.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"

submit_one() {
    local tag="$1"
    local bf16="$2"
    local iters="$3"
    local export_csv="ALL,PROJECT_ROOT=${PROJECT_ROOT},GAZE_MODE=none,ENCODER_LORA_ENABLED=1,ENCODER_LORA_RANK=8,ENCODER_LORA_ALPHA=16.0,ENCODER_LORA_LAST_N_BLOCKS=0,ENCODER_LORA_LR_MULT=0.5,ENCODER_LORA_TARGET_SUFFIXES=attn.qkv,attn.proj,ENCODER_LORA_ACTIVATION_CHECKPOINTING=1,LORA_PROBE_TRAIN_MODE=full,EVAL_SINGLE_PROBE=1,EVAL_LR=0.0001,EVAL_BATCH_SIZE=4,EVAL_GRAD_CLIP=1.0,EVAL_WARMUP_EPOCHS=2,EVAL_MAX_TRAIN_ITERS=${iters},EVAL_NUM_EPOCHS=1,EVAL_NUM_WORKERS=2,EVAL_GRAD_DIAG_INTERVAL=10,EVAL_KEEP_NONFINITE_GRADS=1,EVAL_USE_BFLOAT16=${bf16},LORA_PRETRAINED_PROBE=,LORA_TAG=${tag},CONFIG_PATH=${PROJECT_ROOT}/configs/generated/hdepic_baseline_enclora_graddiag_smoke.yaml,RESUME_CHECKPOINT=0,OUTPUT_DIR=${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation"

    echo "[submit-baseline-keep-nf] tag=${tag} bf16=${bf16} iters=${iters} EVAL_KEEP_NONFINITE_GRADS=1"
    sbatch \
        --partition=h100_tandon \
        --account=your_slurm_account \
        --gres=gpu:h100:1 \
        --cpus-per-task=12 \
        --mem=768GB \
        --time=24:00:00 \
        --job-name="VJEPA2-EXP__baseline_keep_nf" \
        --output="${PROJECT_ROOT}/logs/hdepic_baseline_keep_nonfinite_%j.out" \
        --error="${PROJECT_ROOT}/logs/hdepic_baseline_keep_nonfinite_%j.err" \
        --export="${export_csv}" \
        "${RUN_SCRIPT}"
}

# bf16 is where nonfinite discards were observed; 600 iters ~ early convergence check
submit_one "hdepic-baseline-keep-nonfinite-bf16-i600-w2" 1 600
