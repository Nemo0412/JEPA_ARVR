#!/bin/bash
# Local A6000 launcher for Jepa_PE stream-KV ± probe-RoPE ablation.
#
# Arms (set via env):
#   CACHE_FRAMES=0   PROBE_ROPE=0  → short context baseline (last 16f only)
#   CACHE_FRAMES=112 PROBE_ROPE=0  → KV cache long context
#   CACHE_FRAMES=112 PROBE_ROPE=1  → KV cache + probe temporal RoPE
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 CACHE_FRAMES=0   PROBE_ROPE=0 bash scripts/run_local_stream_kv_ablation.sh
#   CUDA_VISIBLE_DEVICES=1 CACHE_FRAMES=112 PROBE_ROPE=0 bash scripts/run_local_stream_kv_ablation.sh
#   CUDA_VISIBLE_DEVICES=2 CACHE_FRAMES=112 PROBE_ROPE=1 bash scripts/run_local_stream_kv_ablation.sh

set -euo pipefail

CACHE_FRAMES="${CACHE_FRAMES:?Set CACHE_FRAMES=0|16|...|112}"
HORIZON="${HORIZON:-2}"
CHUNK_FRAMES="${CHUNK_FRAMES:-16}"
PROBE_ROPE="${PROBE_ROPE:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-8}"
PATIENCE="${PATIENCE:-3}"
GPU_ID="${GPU_ID:-0}"
# Unique NCCL port per arm/GPU so parallel local jobs do not collide.
MASTER_PORT="${MASTER_PORT:-$((29500 + GPU_ID * 100 + CACHE_FRAMES + PROBE_ROPE))}"
export MASTER_PORT
export MASTER_ADDR="${MASTER_ADDR:-localhost}"

case "${HORIZON}" in 2|6) ;; *) echo "ERROR: HORIZON must be 2 or 6" >&2; exit 2 ;; esac
if (( CACHE_FRAMES % CHUNK_FRAMES != 0 )); then
  echo "ERROR: CACHE_FRAMES must be multiple of CHUNK_FRAMES=${CHUNK_FRAMES}" >&2
  exit 2
fi

PYTHON="${PYTHON:-/mnt/hdd/datasets/HD-EPIC/conda_envs/vjepa/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/lls/workspace/Jepa}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
DATA_ROOT="${DATA_ROOT:-/mnt/hdd/datasets/HD-EPIC}"
EXP_ROOT="${EXP_ROOT:-/mnt/hdd/datasets/HD-EPIC/experiments}"

ANN_DIR="${ANN_DIR:-${DATA_ROOT}/hdepic_vjepa_annotations/clip_split}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/hdepic_vjepa_videos}"
HDEPIC_ANN_ROOT="${HDEPIC_ANN_ROOT:-${DATA_ROOT}/hd-epic-annotations/narrations-and-action-segments}"
CHECKPOINT="${CHECKPOINT:-/mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt}"

export VJEPA_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${TMPDIR:-/mnt/hdd/datasets/HD-EPIC/tmp}"
mkdir -p "${TMPDIR}" "${EXP_ROOT}"

if [[ "${PROBE_ROPE}" == "1" ]]; then
  TAG="clip-stream-kv-rope-c${CACHE_FRAMES}-h${HORIZON}s-vitl16-256-8ep"
  OUTPUT_DIR="${EXP_ROOT}/clip_stream_kv_rope_c${CACHE_FRAMES}_h${HORIZON}s"
else
  TAG="clip-stream-kv-c${CACHE_FRAMES}-h${HORIZON}s-vitl16-256-8ep"
  OUTPUT_DIR="${EXP_ROOT}/clip_stream_kv_matched_c${CACHE_FRAMES}_h${HORIZON}s"
fi
RUN_DIR="${OUTPUT_DIR}/action_anticipation_frozen/${TAG}"
CONFIG_PATH="${RUN_DIR}/config.yaml"
DONE_FLAG="${RUN_DIR}/TRAINING_DONE"
mkdir -p "${RUN_DIR}" "${EXP_ROOT}/logs"

if [[ -f "${DONE_FLAG}" ]]; then
  echo "Already done (${DONE_FLAG}); exiting."
  exit 0
fi

# Optional warm-start (cluster P01 video-joint). Cold-start if missing.
WARM_PROBE="${WARM_PROBE:-}"
WARM_ENC="${WARM_ENC:-}"
RESUME=0
PROBE_CKPT=""
ENCODER_LORA_CKPT=""
if [[ -f "${RUN_DIR}/latest.pt" ]]; then
  RESUME=1
  PROBE_CKPT="${RUN_DIR}/latest.pt"
  ENCODER_LORA_CKPT="${RUN_DIR}/encoder_lora_latest.pt"
elif [[ -n "${WARM_PROBE}" && -f "${WARM_PROBE}" ]]; then
  PROBE_CKPT="${WARM_PROBE}"
  ENCODER_LORA_CKPT="${WARM_ENC}"
fi

for f in "${CHECKPOINT}" \
         "${ANN_DIR}/HD_EPIC_train_vjepa.csv" "${ANN_DIR}/HD_EPIC_val_vjepa.csv" \
         "${HDEPIC_ANN_ROOT}/HD_EPIC_Narrations.pkl"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done
[[ -x "${PYTHON}" ]] || { echo "ERROR: bad PYTHON=${PYTHON}" >&2; exit 1; }

cat > "${RUN_DIR}/protocol.json" <<EOF
{
  "dataset": "clip_split HD-EPIC P01 (local)",
  "protocol": "stream_kv_matched_train_eval",
  "cache_frames": ${CACHE_FRAMES},
  "chunk_frames": ${CHUNK_FRAMES},
  "window_frames": $((CACHE_FRAMES + CHUNK_FRAMES)),
  "horizon_sec": ${HORIZON},
  "frames_per_clip_loaded": 128,
  "probe_rope": $([[ "${PROBE_ROPE}" == "1" ]] && echo true || echo false),
  "machine": "$(hostname)",
  "note": "Local ablation: KV cache length vs probe temporal RoPE. no_predictor."
}
EOF

echo "========================================================"
echo "LOCAL stream KV  CACHE=${CACHE_FRAMES}f + chunk=${CHUNK_FRAMES}f  HORIZON=${HORIZON}s  RoPE=${PROBE_ROPE}"
echo "GPU=${GPU_ID}  MASTER_PORT=${MASTER_PORT}  OUT=${RUN_DIR}"
echo "Start: $(date)"
echo "========================================================"
cat "${RUN_DIR}/protocol.json"

"${PYTHON}" - <<PY
import sys, yaml
from pathlib import Path
sys.path.insert(0, "${PROJECT_ROOT}")
sys.path.insert(0, "${VJEPA_ROOT}")
horizon = float("${HORIZON}")
cache_frames = int("${CACHE_FRAMES}")
chunk_frames = int("${CHUNK_FRAMES}")
probe_rope = bool(int("${PROBE_ROPE}"))
resume = bool(int("${RESUME}"))
batch_size = int("${BATCH_SIZE}")
probe_ckpt = "${PROBE_CKPT}"
enc_lora_ckpt = "${ENCODER_LORA_CKPT}"
template_path = Path("${VJEPA_ROOT}/configs/eval/vitl/ek100.yaml")
config_path = Path("${CONFIG_PATH}")
with template_path.open() as f:
    cfg = yaml.safe_load(f)
cfg["nodes"] = 1
cfg["tasks_per_node"] = 1
cfg["cpus_per_task"] = 8
cfg["tag"] = "${TAG}"
cfg["eval_name"] = "app.hdepic_lora_action_anticipation"
cfg["folder"] = "${OUTPUT_DIR}"
cfg["resume_checkpoint"] = resume
cfg["val_only"] = False
data = cfg["experiment"]["data"]
num_workers = int("${NUM_WORKERS}")
val_num_workers = int("${VAL_NUM_WORKERS}")
prefetch_factor = int("${PREFETCH_FACTOR}")
data.update({
    "dataset": "EK100", "file_format": 1,
    "base_path": "${VIDEO_ROOT}",
    "dataset_train": "${ANN_DIR}/HD_EPIC_train_vjepa.csv",
    "dataset_val": "${ANN_DIR}/HD_EPIC_val_vjepa.csv",
    "num_workers": num_workers,
    "val_num_workers": val_num_workers,
    "clip_balanced": True,
    "prefetch_factor": prefetch_factor, "pin_memory": False,
    "persistent_workers": bool(num_workers > 0),
    "resolution": 256, "frames_per_clip": 128, "frames_per_second": 8,
    "anticipation_time_sec": [horizon, horizon],
    "train_anticipation_time_sec": [horizon, horizon],
})
opt = cfg["experiment"]["optimization"]
opt.update({
    "batch_size": batch_size, "num_epochs": int("${NUM_EPOCHS}"), "use_bfloat16": True,
    "use_focal_loss": False, "val_every_epochs": 1, "best_metric": "val-action-top5",
    "early_stopping_patience": int("${PATIENCE}"), "grad_clip": 1.0,
    "multihead_kwargs": [{
        "weight_decay": 0.0001, "final_weight_decay": 0.0001,
        "lr": 0.00002, "start_lr": 0.0, "final_lr": 0.0, "warmup": 1,
    }],
})
cfg["model_kwargs"]["checkpoint"] = "${CHECKPOINT}"
cfg["model_kwargs"]["module_name"] = "evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar"
cfg["model_kwargs"]["wrapper_kwargs"] = {
    "no_predictor": True,
    "num_output_frames": 2,
    "num_steps": 1,
}
cfg["experiment"]["classifier"] = {"num_probe_blocks": 4, "num_heads": 16}
lora = {
    "enabled": True, "rank": 8, "alpha": 16.0, "dropout": 0.05,
    "probe_train_mode": "full", "train_heads": True, "freeze_pooler": False,
    "align_reference_metrics": True, "val_metric_scope": "native",
    "val_metric_aggregation": "metric_wise_max", "class_space": "train_only",
    "temporal_sampling": "phd_reference",
    "output_regularization": {"enabled": False},
    "hdepic_reference": {
        "annotations_pkl": "${HDEPIC_ANN_ROOT}/HD_EPIC_Narrations.pkl",
        "verb_classes_csv": "${HDEPIC_ANN_ROOT}/HD_EPIC_verb_classes.csv",
        "noun_classes_csv": "${HDEPIC_ANN_ROOT}/HD_EPIC_noun_classes.csv",
    },
    "gaze": {"mode": "none"},
    "stream_kv": {
        "enabled": True,
        "cache_frames": cache_frames,
        "chunk_frames": chunk_frames,
        "train_last_chunk_only": True,
    },
    "probe_temporal_rope": {
        "enabled": probe_rope,
        # Match upstream Jepa_PE: RoPE on Probe.blocks[0] only, via frame/slot ids.
        "only_block0": True,
        "rope_cross_attn_k": False,
        "grid_size": 16,
        "tubelet_size": 2,
    },
    "encoder_lora": {
        "enabled": True, "rank": 8, "alpha": 16.0, "dropout": 0.05,
        "last_n_blocks": 0, "lr_mult": 0.5, "weight_decay": 0.0001,
        "activation_checkpointing": False,
        "target_suffixes": ["attn.qkv", "attn.proj"],
        "warm_start_at_init": True, "freeze": False,
        "checkpoint_path": "${RUN_DIR}/encoder_lora_latest.pt",
    },
    "predictor_lora": {"enabled": False},
}
if probe_ckpt:
    lora["pretrained_probe"] = probe_ckpt
    lora["load_probe_heads"] = True
if enc_lora_ckpt and Path(enc_lora_ckpt).is_file():
    lora["encoder_lora"]["load_checkpoint_path"] = enc_lora_ckpt
cfg["experiment"]["lora"] = lora
config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(
    f"Config: {config_path} stream_kv cache={cache_frames} chunk={chunk_frames} "
    f"probe_rope={probe_rope} horizon={horizon}s resume={resume} bs={batch_size} "
    f"warm_probe={bool(probe_ckpt)}"
)
PY

cd "${VJEPA_ROOT}"
LOG="${EXP_ROOT}/logs/${TAG}_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${LOG}"
# Do NOT preset CUDA_VISIBLE_DEVICES: evals.main process_main remaps from --devices.
# Pass a single device so world_size=1 and NCCL init uses MASTER_PORT above.
set +e
"${PYTHON}" -m evals.main --fname "${CONFIG_PATH}" --devices "cuda:${GPU_ID}" 2>&1 | tee "${LOG}"
train_rc=${PIPESTATUS[0]}
set -e

echo "Training exit_code=${train_rc} CACHE=${CACHE_FRAMES} HORIZON=${HORIZON}s RoPE=${PROBE_ROPE} at $(date)"
exit "${train_rc}"
