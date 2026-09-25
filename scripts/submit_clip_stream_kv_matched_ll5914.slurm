#!/bin/bash
#SBATCH --job-name=kvmatch
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:50:00
#SBATCH --signal=B:USR1@90
#SBATCH --output=/scratch/ll5914/logs/kvmatch_%j.out
#SBATCH --error=/scratch/ll5914/logs/kvmatch_%j.err

set -euo pipefail

# Train/eval matched stream KV (newest-aligned chunked cache).
# Same protocol as eval_nopred_kvcache_size_sweep — not one-shot full128.
#
#   CACHE_FRAMES=0    → only last 2s (16f)   = short context
#   CACHE_FRAMES=112  → 14s history + 2s     = long context
#
# Expectation under matched train+eval: long ≥ short (more frames help).
#
# Prefer low mem (96G) + 3-partition race (A100/H100/H200); first starter
# cancels same-name siblings. Probe RoPE defaults to blocks[0] only.
#
# Submit example:
#   EXPORT=CACHE_FRAMES=112,HORIZON=2,BATCH_SIZE=2,NUM_WORKERS=2,VAL_NUM_WORKERS=1,PREFETCH_FACTOR=1,PROBE_ROPE=0
#   for spec in 'a100_tandon|gpu:a100:1' 'h100_tandon|gpu:h100:1' 'h200_tandon|gpu:h200:1'; do
#     IFS='|' read -r part gres <<< "$spec"
#     sbatch --job-name=kvmatch112 --partition="$part" --gres="$gres" \
#       --mem=96G --cpus-per-task=8 --export="$EXPORT" this.slurm
#   done

CACHE_FRAMES="${CACHE_FRAMES:?Set CACHE_FRAMES=0|16|...|112}"
HORIZON="${HORIZON:-2}"
CHUNK_FRAMES="${CHUNK_FRAMES:-16}"
PROBE_ROPE="${PROBE_ROPE:-0}"
case "${HORIZON}" in 2|6) ;; *) echo "ERROR: HORIZON must be 2 or 6" >&2; exit 2 ;; esac
if (( CACHE_FRAMES % CHUNK_FRAMES != 0 )); then
  echo "ERROR: CACHE_FRAMES must be multiple of CHUNK_FRAMES=${CHUNK_FRAMES}" >&2
  exit 2
fi

BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export MID_EPOCH_SAVE_EVERY="${MID_EPOCH_SAVE_EVERY:-50}"

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/submit_clip_stream_kv_matched_ll5914.slurm"
export VJEPA_ROOT PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${TMPDIR:-/scratch/ll5914/tmp}"
mkdir -p "${TMPDIR}"

ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split"
VIDEO_ROOT_REMOTE="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
HDEPIC_ANN_ROOT="/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"
VIDEO_BEST_DIR="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep"
# Prefer continuing from matched stream-KV (no-RoPE) weights when adding RoPE.
KV112_RUN="/scratch/ll5914/experiments/clip_stream_kv_matched_c112_h2s/action_anticipation_frozen/clip-stream-kv-c112-h2s-vitl16-256-8ep"

if [[ "${PROBE_ROPE}" == "1" ]]; then
  TAG="clip-stream-kv-rope-c${CACHE_FRAMES}-h${HORIZON}s-vitl16-256-8ep"
  OUTPUT_DIR="/scratch/ll5914/experiments/clip_stream_kv_rope_c${CACHE_FRAMES}_h${HORIZON}s"
else
  TAG="clip-stream-kv-c${CACHE_FRAMES}-h${HORIZON}s-vitl16-256-8ep"
  OUTPUT_DIR="/scratch/ll5914/experiments/clip_stream_kv_matched_c${CACHE_FRAMES}_h${HORIZON}s"
fi
RUN_DIR="${OUTPUT_DIR}/action_anticipation_frozen/${TAG}"
CONFIG_PATH="${RUN_DIR}/config.yaml"
DONE_FLAG="${RUN_DIR}/TRAINING_DONE"
NUM_EPOCHS=8
PATIENCE=3
mkdir -p /scratch/ll5914/logs "${OUTPUT_DIR}" "${RUN_DIR}"

if [[ -f "${DONE_FLAG}" ]]; then
  echo "Already done (${DONE_FLAG}); exiting."
  exit 0
fi

LOCK="${RUN_DIR}/RUNNING_JOB"
if [[ -f "${LOCK}" ]]; then
  other="$(cat "${LOCK}" 2>/dev/null || true)"
  if [[ -n "${other}" && "${other}" != "${SLURM_JOB_ID:-}" ]] && squeue -j "${other}" -h -t RUNNING 2>/dev/null | grep -q .; then
    echo "RUN_DIR already owned by running job ${other}; exiting."
    exit 0
  fi
fi
echo "${SLURM_JOB_ID:-manual}" > "${LOCK}"

# Multi-partition race: first to start cancels same-name siblings on other GPUs.
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NAME:-}" ]]; then
  while read -r sib; do
    [[ -z "${sib}" || "${sib}" == "${SLURM_JOB_ID}" ]] && continue
    echo "Canceling sibling ${sib} (same name ${SLURM_JOB_NAME})"
    scancel --signal=KILL "${sib}" 2>/dev/null || true
  done < <(squeue -u "${USER}" -n "${SLURM_JOB_NAME}" -h -o '%i' 2>/dev/null || true)
fi

# No /dev/shm stage: read P01 symlink farm from scratch.
VIDEO_ROOT="${VIDEO_ROOT_REMOTE}"
_RESUBMITTED=0
TRAIN_PID=""

request_resubmit() {
  if [[ -f "${DONE_FLAG}" || "${_RESUBMITTED}" == "1" ]]; then
    return 0
  fi
  _RESUBMITTED=1
  echo "Auto-resubmitting CACHE_FRAMES=${CACHE_FRAMES} HORIZON=${HORIZON} after job ${SLURM_JOB_ID:-}..."
  dep=()
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    dep=(--dependency="afterany:${SLURM_JOB_ID}")
  fi
  # Keep TMPDIR on scratch — auto-resubmit previously failed silently when /tmp was full.
  export TMPDIR="${TMPDIR:-/scratch/ll5914/tmp}"
  mkdir -p "${TMPDIR}"
  # Do NOT reuse SLURM_JOB_GRES (often "gpu:1" without type → invalid on tandon).
  # Race A100/H100/H200 with explicit typed gres + low mem.
  local mem_gb="${SLURM_MEM_PER_NODE:-96G}"
  # SLURM_MEM_PER_NODE may be numeric MB; normalize to G if needed.
  if [[ "${mem_gb}" =~ ^[0-9]+$ ]]; then
    mem_gb="$((mem_gb / 1024))G"
  fi
  local ok=0
  for spec in 'a100_tandon|gpu:a100:1' 'h100_tandon|gpu:h100:1' 'h200_tandon|gpu:h200:1'; do
    IFS='|' read -r part gres <<< "$spec"
    out="$(sbatch --job-name="${SLURM_JOB_NAME:-kvmatch}" \
      --account="${SLURM_JOB_ACCOUNT:-torch_pr_111_tandon_advanced}" \
      --partition="$part" \
      --gres="$gres" \
      --mem="${mem_gb}" \
      --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
      --time="${SLURM_TIMELIMIT:-01:50:00}" \
      "${dep[@]}" \
      --export=CACHE_FRAMES="${CACHE_FRAMES}",HORIZON="${HORIZON}",CHUNK_FRAMES="${CHUNK_FRAMES}",BATCH_SIZE="${BATCH_SIZE}",NUM_WORKERS="${NUM_WORKERS}",VAL_NUM_WORKERS="${VAL_NUM_WORKERS}",PREFETCH_FACTOR="${PREFETCH_FACTOR}",PROBE_ROPE="${PROBE_ROPE}" \
      "${SCRIPT_PATH}" 2>&1)" || true
    echo "$part => ${out}"
    if grep -q 'Submitted batch job' <<<"${out}"; then
      ok=1
    fi
  done
  if [[ "${ok}" != "1" ]]; then
    echo "WARN: sbatch resubmit failed on all partitions" >&2
  fi
}

on_signal() {
  echo "Caught signal — save then resubmit @ $(date)"
  request_resubmit
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" 2>/dev/null || true
    TRAIN_PID=""
  fi
}

trap on_signal USR1 TERM

WARM_PROBE="${VIDEO_BEST_DIR}/best.pt"
WARM_ENC="${VIDEO_BEST_DIR}/encoder_lora_best.pt"
if [[ "${PROBE_ROPE}" == "1" && -f "${KV112_RUN}/latest.pt" ]]; then
  WARM_PROBE="${KV112_RUN}/latest.pt"
  WARM_ENC="${KV112_RUN}/encoder_lora_latest.pt"
  echo "RoPE warm-start from stream-KV matched c112: ${KV112_RUN}"
fi
RESUME=0
PROBE_CKPT="${WARM_PROBE}"
ENCODER_LORA_CKPT="${WARM_ENC}"
if [[ -f "${RUN_DIR}/latest.pt" ]]; then
  RESUME=1
  PROBE_CKPT="${RUN_DIR}/latest.pt"
  ENCODER_LORA_CKPT="${RUN_DIR}/encoder_lora_latest.pt"
fi

for f in "${CHECKPOINT}" "${PROBE_CKPT}" "${ENCODER_LORA_CKPT}" \
         "${ANN_DIR}/HD_EPIC_train_vjepa.csv" "${ANN_DIR}/HD_EPIC_val_vjepa.csv"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done

export MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))

cat > "${RUN_DIR}/protocol.json" <<EOF
{
  "dataset": "clip_split HD-EPIC P01",
  "protocol": "stream_kv_matched_train_eval",
  "cache_frames": ${CACHE_FRAMES},
  "chunk_frames": ${CHUNK_FRAMES},
  "window_frames": $((CACHE_FRAMES + CHUNK_FRAMES)),
  "horizon_sec": ${HORIZON},
  "frames_per_clip_loaded": 128,
  "note": "Data loads 128f newest-aligned; encode uses last cache+chunk via streamed KV"
}
EOF

echo "========================================================"
echo "stream KV matched  CACHE=${CACHE_FRAMES}f + chunk=${CHUNK_FRAMES}f  HORIZON=${HORIZON}s"
echo "Job=${SLURM_JOB_ID:-manual} resume=${RESUME} OUT=${RUN_DIR}"
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
template_path = Path("${VJEPA_ROOT}/configs/eval/vitl/ek100.yaml")
config_path = Path("${CONFIG_PATH}")
with template_path.open() as f:
    cfg = yaml.safe_load(f)
cfg["nodes"] = 1
cfg["tasks_per_node"] = 1
cfg["cpus_per_task"] = int("${SLURM_CPUS_PER_TASK:-24}")
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
    "pretrained_probe": "${PROBE_CKPT}",
    "load_probe_heads": True,
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
        "only_block0": False,
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
        "load_checkpoint_path": "${ENCODER_LORA_CKPT}",
        "checkpoint_path": "${RUN_DIR}/encoder_lora_latest.pt",
    },
    "predictor_lora": {"enabled": False},
}
cfg["experiment"]["lora"] = lora
config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(
    f"Config: {config_path} stream_kv cache={cache_frames} chunk={chunk_frames} "
    f"probe_rope={probe_rope} horizon={horizon}s resume={resume} bs={batch_size}"
)
PY

cd "${VJEPA_ROOT}"
set +e
"${PYTHON}" -m evals.main --fname "${CONFIG_PATH}" --devices cuda:0 &
TRAIN_PID=$!
wait "${TRAIN_PID}"
train_rc=$?
TRAIN_PID=""
set -e

echo "Training exit_code=${train_rc} CACHE=${CACHE_FRAMES} HORIZON=${HORIZON}s at $(date)"
"${PYTHON}" - <<'PY'
from pathlib import Path
run = Path("${RUN_DIR}")
done = run / "TRAINING_DONE"
latest = run / "latest.pt"
print(f"latest exists={latest.exists()} done={done.is_file()}")
print(f"NEED_RESUBMIT={0 if done.is_file() else 1}")
PY

if [[ ! -f "${DONE_FLAG}" ]]; then
  request_resubmit
fi
exit "${train_rc}"
