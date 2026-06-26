#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_probe
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_probe_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_probe_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Sidecar LoRA finetune path for the action anticipation probe/decoder.
# Does not edit vjepa2/. Uses eval_name=app.hdepic_lora_action_anticipation.
# Default path uses the official EK100 ViT-g/384 probe as the decoder starting
# point because its pooler dimensions match the official vitg-384 encoder.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

ADAPTED_ANNOTATIONS_DIR="${ADAPTED_ANNOTATIONS_DIR:-${PROJECT_ROOT}/data/hdepic_vjepa_annotations}"
ADAPTED_VIDEO_ROOT="${ADAPTED_VIDEO_ROOT:-${PROJECT_ROOT}/data/hdepic_vjepa_videos}"
CONFIG_DIR="${CONFIG_DIR:-${PROJECT_ROOT}/configs/generated}"
CONFIG_PATH="${CONFIG_PATH:-${CONFIG_DIR}/hdepic_lora_probe_vitg384.yaml}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vitg-384.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"
LORA_PRETRAINED_PROBE="${LORA_PRETRAINED_PROBE-${PROJECT_ROOT}/outputs/ek100_vitg384_probe/action_anticipation_frozen/ek100-vitg16-384/latest.pt}"
LORA_TAG="${LORA_TAG:-hdepic-lora-vitg16-384-cleanp01-refmetrics}"
GAZE_MODE="${GAZE_MODE:-none}"
SIGMA="${SIGMA:-96}"
GAZE_VIDEO_ROOT="${GAZE_VIDEO_ROOT:-${PROJECT_ROOT}/data/hdepic_vjepa_videos_gaze_sigma${SIGMA}}"
GAZE_ROOT="${GAZE_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/SLAM-and-Gaze}"
GAZE_EXTRACT_ROOT="${GAZE_EXTRACT_ROOT:-${PROJECT_ROOT}/outputs/hdepic_gaze_token_gate/extracted}"
GAZE_SYNC_ROOT="${GAZE_SYNC_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/Videos}"
GAZE_GATE_GAMMA="${GAZE_GATE_GAMMA:-0.7}"
GAZE_GATE_SIGMA_PX="${GAZE_GATE_SIGMA_PX:-40}"
GAZE_LEARNABLE_GATE="${GAZE_LEARNABLE_GATE:-1}"
GAZE_RNN_HIDDEN_DIM="${GAZE_RNN_HIDDEN_DIM:-256}"
GAZE_RNN_NUM_LAYERS="${GAZE_RNN_NUM_LAYERS:-2}"
GAZE_RNN_NUM_TOKENS="${GAZE_RNN_NUM_TOKENS:-64}"
GAZE_RNN_DROPOUT="${GAZE_RNN_DROPOUT:-0.1}"
GAZE_RNN_BIDIR="${GAZE_RNN_BIDIR:-1}"
GAZE_RNN_TRAJ_LEN="${GAZE_RNN_TRAJ_LEN:-1024}"
GAZE_RNN_USE_VIDEO_TOKEN="${GAZE_RNN_USE_VIDEO_TOKEN:-0}"
GAZE_RNN_VIDEO_FUSION="${GAZE_RNN_VIDEO_FUSION:-nearest_concat}"
GAZE_RNN_VIDEO_PROJ_DIM="${GAZE_RNN_VIDEO_PROJ_DIM:-128}"
GAZE_RNN_LOCAL_TEMPORAL_RADIUS="${GAZE_RNN_LOCAL_TEMPORAL_RADIUS:-0}"
GAZE_RNN_LOCAL_SPATIAL_RADIUS="${GAZE_RNN_LOCAL_SPATIAL_RADIUS:-1}"
GAZE_RNN_RESIDUAL_ALPHA_INIT="${GAZE_RNN_RESIDUAL_ALPHA_INIT:-0.01}"
GAZE_RNN_HISTORY_SEC="${GAZE_RNN_HISTORY_SEC:-0}"
GAZE_LR_MULT="${GAZE_LR_MULT:-5.0}"
GAZE_HIDDEN_DUMP="${GAZE_HIDDEN_DUMP:-0}"
PREDICTION_DUMP="${PREDICTION_DUMP:-0}"
PREDICTION_DUMP_PATH="${PREDICTION_DUMP_PATH:-}"
LORA_VAL_METRIC_SCOPE="${LORA_VAL_METRIC_SCOPE:-native}"
# Debug-subset path (default empty = disabled). Set to configs/debug_subset_p01.json to restrict
# both train and val to the fixed P01 debug subset. Always include DEBUG_SUBSET in LORA_TAG when
# using this; results are not valid for teacher-facing reporting.
DEBUG_SUBSET_PATH="${DEBUG_SUBSET_PATH:-}"
GAZE_DATALOADER_TRACE_DIR="${GAZE_DATALOADER_TRACE_DIR:-}"
GAZE_DATALOADER_TRACE_EVENTS="${GAZE_DATALOADER_TRACE_EVENTS:-0}"
PERF_MONITOR="${PERF_MONITOR:-0}"
PERF_INTERVAL="${PERF_INTERVAL:-30}"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_PROBE_TRAIN_MODE="${LORA_PROBE_TRAIN_MODE:-lora}"
LORA_TRAIN_HEADS="${LORA_TRAIN_HEADS:-1}"
EVAL_GPUS="${EVAL_GPUS:-1}"
EVAL_DEBUGMODE="${EVAL_DEBUGMODE:-auto}"
EVAL_CPUS_PER_TASK="${EVAL_CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-12}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-}"
EVAL_MAX_TRAIN_ITERS="${EVAL_MAX_TRAIN_ITERS:-0}"
EVAL_MAX_LR="${EVAL_MAX_LR:-}"
EVAL_LR_SCALE="${EVAL_LR_SCALE:-1}"
EVAL_SINGLE_PROBE="${EVAL_SINGLE_PROBE:-0}"
EVAL_LR="${EVAL_LR:-}"
EVAL_START_LR="${EVAL_START_LR:-}"
EVAL_FINAL_LR="${EVAL_FINAL_LR:-}"
EVAL_WARMUP_EPOCHS="${EVAL_WARMUP_EPOCHS:-}"
EVAL_WEIGHT_DECAY="${EVAL_WEIGHT_DECAY:-}"
EVAL_FINAL_WEIGHT_DECAY="${EVAL_FINAL_WEIGHT_DECAY:-}"
EVAL_USE_FOCAL_LOSS="${EVAL_USE_FOCAL_LOSS:-}"
EVAL_GRAD_CLIP="${EVAL_GRAD_CLIP:-0}"
EVAL_USE_BFLOAT16="${EVAL_USE_BFLOAT16:-}"
EVAL_BEST_METRIC="${EVAL_BEST_METRIC:-val-action-acc}"
EVAL_EARLY_STOPPING_PATIENCE="${EVAL_EARLY_STOPPING_PATIENCE:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"
VAL_ONLY="${VAL_ONLY:-0}"
BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP="${BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP:-0}"
BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT="${BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT:-}"
BINARY_INPUT_ADAPTER_AUG_AWARE="${BINARY_INPUT_ADAPTER_AUG_AWARE:-0}"
GAZE_BINARY_MAP_TYPE="${GAZE_BINARY_MAP_TYPE:-binary}"
GAZE_DECODER_CACHE_READER="${GAZE_DECODER_CACHE_READER:-1}"
GAZE_DECODER_NUM_THREADS="${GAZE_DECODER_NUM_THREADS:-1}"
ENCODER_GAZE_ADAPTER_NUM_HEADS="${ENCODER_GAZE_ADAPTER_NUM_HEADS:-4}"
ENCODER_GAZE_ADAPTER_DROPOUT="${ENCODER_GAZE_ADAPTER_DROPOUT:-0.0}"
ENCODER_GAZE_ADAPTER_LR_MULT="${ENCODER_GAZE_ADAPTER_LR_MULT:-0.05}"
ENCODER_GAZE_ADAPTER_WEIGHT_DECAY="${ENCODER_GAZE_ADAPTER_WEIGHT_DECAY:-0.0001}"
ENCODER_LORA_ENABLED="${ENCODER_LORA_ENABLED:-0}"
ENCODER_LORA_RANK="${ENCODER_LORA_RANK:-${LORA_RANK}}"
ENCODER_LORA_ALPHA="${ENCODER_LORA_ALPHA:-${LORA_ALPHA}}"
ENCODER_LORA_DROPOUT="${ENCODER_LORA_DROPOUT:-${LORA_DROPOUT}}"
ENCODER_LORA_LAST_N_BLOCKS="${ENCODER_LORA_LAST_N_BLOCKS:-12}"
ENCODER_LORA_LR_MULT="${ENCODER_LORA_LR_MULT:-0.5}"
ENCODER_LORA_WEIGHT_DECAY="${ENCODER_LORA_WEIGHT_DECAY:-0.0001}"
ENCODER_LORA_ACTIVATION_CHECKPOINTING="${ENCODER_LORA_ACTIVATION_CHECKPOINTING:-0}"
ENCODER_LORA_TARGET_SUFFIXES="${ENCODER_LORA_TARGET_SUFFIXES:-}"
HDEPIC_DIRECT_ROPE_10S="${HDEPIC_DIRECT_ROPE_10S:-0}"
MODEL_ROPE_SCALE_MODE="${MODEL_ROPE_SCALE_MODE:-ntk_temporal}"
EVAL_ANTICIPATION_SEC="${EVAL_ANTICIPATION_SEC:-}"
EVAL_TRAIN_ANTICIPATION_SEC_MIN="${EVAL_TRAIN_ANTICIPATION_SEC_MIN:-}"
EVAL_TRAIN_ANTICIPATION_SEC_MAX="${EVAL_TRAIN_ANTICIPATION_SEC_MAX:-}"
MODEL_RETURN_MODE="${MODEL_RETURN_MODE:-}"
MODEL_MAX_ROLLOUT_STEPS="${MODEL_MAX_ROLLOUT_STEPS:-512}"
EVAL_VAL_EVERY="${EVAL_VAL_EVERY:-}"
MODEL_FAMILY="${MODEL_FAMILY:-vjepa}"
VLM_MODEL_ID="${VLM_MODEL_ID:-}"
VLM_MODEL_CLASS="${VLM_MODEL_CLASS:-auto}"
VLM_NUM_FRAMES="${VLM_NUM_FRAMES:-8}"
VLM_TOKEN_MODE="${VLM_TOKEN_MODE:-pooled}"
VLM_IMAGE_SIZE="${VLM_IMAGE_SIZE:-${EVAL_RESOLUTION:-224}}"
VLM_TRUST_REMOTE_CODE="${VLM_TRUST_REMOTE_CODE:-0}"
VLM_TORCH_DTYPE="${VLM_TORCH_DTYPE:-}"
VLM_LOCAL_FILES_ONLY="${VLM_LOCAL_FILES_ONLY:-0}"
VLM_GPU_PULSE_ITERS="${VLM_GPU_PULSE_ITERS:-0}"
VLM_GPU_PULSE_SIZE="${VLM_GPU_PULSE_SIZE:-512}"

mkdir -p "${PROJECT_ROOT}/logs" "${CONFIG_DIR}" "${OUTPUT_DIR}"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export ADAPTED_ANNOTATIONS_DIR ADAPTED_VIDEO_ROOT CONFIG_DIR CONFIG_PATH CHECKPOINT OUTPUT_DIR LORA_PRETRAINED_PROBE
export LORA_TAG LORA_RANK LORA_ALPHA LORA_DROPOUT LORA_PROBE_TRAIN_MODE LORA_TRAIN_HEADS LORA_VAL_METRIC_SCOPE EVAL_GPUS EVAL_DEBUGMODE EVAL_CPUS_PER_TASK EVAL_BATCH_SIZE EVAL_NUM_EPOCHS EVAL_NUM_WORKERS EVAL_VAL_NUM_WORKERS EVAL_MAX_TRAIN_ITERS EVAL_MAX_LR EVAL_LR_SCALE EVAL_SINGLE_PROBE EVAL_LR EVAL_START_LR EVAL_FINAL_LR EVAL_WARMUP_EPOCHS EVAL_WEIGHT_DECAY EVAL_FINAL_WEIGHT_DECAY EVAL_USE_FOCAL_LOSS EVAL_GRAD_CLIP EVAL_USE_BFLOAT16 EVAL_BEST_METRIC EVAL_EARLY_STOPPING_PATIENCE RESUME_CHECKPOINT VAL_ONLY
export GAZE_MODE SIGMA GAZE_VIDEO_ROOT GAZE_ROOT GAZE_EXTRACT_ROOT GAZE_SYNC_ROOT GAZE_GATE_GAMMA GAZE_GATE_SIGMA_PX GAZE_LEARNABLE_GATE
export GAZE_BINARY_MAP_TYPE GAZE_DECODER_CACHE_READER GAZE_DECODER_NUM_THREADS
export GAZE_RNN_HIDDEN_DIM GAZE_RNN_NUM_LAYERS GAZE_RNN_NUM_TOKENS GAZE_RNN_DROPOUT GAZE_RNN_BIDIR GAZE_RNN_TRAJ_LEN
export GAZE_RNN_USE_VIDEO_TOKEN GAZE_RNN_VIDEO_FUSION GAZE_RNN_VIDEO_PROJ_DIM GAZE_RNN_LOCAL_TEMPORAL_RADIUS GAZE_RNN_LOCAL_SPATIAL_RADIUS GAZE_RNN_RESIDUAL_ALPHA_INIT GAZE_RNN_HISTORY_SEC GAZE_LR_MULT GAZE_HIDDEN_DUMP
export POSE_SLAM_ROOT POSE_MAPPING_JSON POSE_FEATURE_SET POSE_QUALITY_MIN POSE_HISTORY_SEC
export BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT BINARY_INPUT_ADAPTER_AUG_AWARE
export ENCODER_GAZE_ADAPTER_NUM_HEADS ENCODER_GAZE_ADAPTER_DROPOUT ENCODER_GAZE_ADAPTER_LR_MULT ENCODER_GAZE_ADAPTER_WEIGHT_DECAY
export ENCODER_LORA_ENABLED ENCODER_LORA_RANK ENCODER_LORA_ALPHA ENCODER_LORA_DROPOUT ENCODER_LORA_LAST_N_BLOCKS ENCODER_LORA_LR_MULT ENCODER_LORA_WEIGHT_DECAY ENCODER_LORA_ACTIVATION_CHECKPOINTING ENCODER_LORA_TARGET_SUFFIXES
export BACKBONE EVAL_RESOLUTION AR10S_SLIDING_PROBE HDEPIC_AR10S_SLIDING_GAZE_POSE HDEPIC_DIRECT_ROPE_10S MODEL_ROPE_SCALE_MODE EVAL_ANTICIPATION_SEC EVAL_TRAIN_ANTICIPATION_SEC_MIN EVAL_TRAIN_ANTICIPATION_SEC_MAX MODEL_RETURN_MODE MODEL_MAX_ROLLOUT_STEPS EVAL_VAL_EVERY
export MODEL_FAMILY VLM_MODEL_ID VLM_MODEL_CLASS VLM_NUM_FRAMES VLM_TOKEN_MODE VLM_IMAGE_SIZE VLM_TRUST_REMOTE_CODE VLM_TORCH_DTYPE VLM_LOCAL_FILES_ONLY VLM_GPU_PULSE_ITERS VLM_GPU_PULSE_SIZE
export PREDICTION_DUMP PREDICTION_DUMP_PATH
export DEBUG_SUBSET_PATH GAZE_DATALOADER_TRACE_DIR GAZE_DATALOADER_TRACE_EVENTS
export PERF_MONITOR PERF_INTERVAL

payload=""
read -r -d '' payload <<'BASH' || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

case "${GAZE_MODE}" in
  none|token_gate|rnn_fuse|mlp_fuse|pose_rnn_fuse|multimodal_rnn_fuse|binary_input_adapter|binary_input_adapter_pose_rnn_fuse|binary_input_adapter_gaze_pose_matrix|encoder_output_inject)
    ;;
  overlay_video)
    if [[ ! -d "${GAZE_VIDEO_ROOT}" ]]; then
      echo "[lora] Missing gaze overlay video root: ${GAZE_VIDEO_ROOT}" >&2
      echo "[lora] Run scripts/prepare_hdepic_gaze_val_tree_cpu.slurm or create a train/val overlay tree first." >&2
      exit 1
    fi
    export ADAPTED_VIDEO_ROOT="${GAZE_VIDEO_ROOT}"
    ;;
  *)
    echo "[lora] Unknown GAZE_MODE=${GAZE_MODE}; expected none, overlay_video, token_gate, rnn_fuse, mlp_fuse, pose_rnn_fuse, multimodal_rnn_fuse, binary_input_adapter, binary_input_adapter_pose_rnn_fuse, binary_input_adapter_gaze_pose_matrix, or encoder_output_inject" >&2
    exit 1
    ;;
esac

if [[ ! -f "${PROJECT_ROOT}/app/hdepic_lora_action_anticipation/eval.py" ]]; then
    echo "[lora] Missing ${PROJECT_ROOT}/app/hdepic_lora_action_anticipation/eval.py" >&2
    echo "[lora] Sync the project-level app/ directory to HPC before running this script." >&2
    exit 1
fi

python - <<'PY'
import importlib
import os
import sys

print("[lora] sys.path head:", sys.path[:5])
importlib.import_module("app.hdepic_lora_action_anticipation.eval")
print("[lora] app.hdepic_lora_action_anticipation import OK")
importlib.import_module("app.hdepic_lora_action_anticipation.encoder_lora")
print("[lora] encoder_lora import OK")

def _truthy(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}

if _truthy("HDEPIC_DIRECT_ROPE_10S"):
    importlib.import_module("app.hdepic_lora_action_anticipation.modelcustom.vit_encoder_predictor_direct_rope")
    print("[lora] vit_encoder_predictor_direct_rope import OK")
elif os.environ.get("MODEL_FAMILY", "vjepa").strip().lower() == "vlm":
    importlib.import_module("app.hdepic_lora_action_anticipation.modelcustom.vlm_video_encoder")
    print("[lora] vlm_video_encoder import OK")
elif _truthy("AR10S_SLIDING_PROBE") or _truthy("HDEPIC_AR10S_SLIDING_GAZE_POSE") or os.environ.get("EVAL_ANTICIPATION_SEC", "").strip():
    importlib.import_module("app.hdepic_lora_action_anticipation.modelcustom.vit_encoder_predictor_rollout")
    print("[lora] vit_encoder_predictor_rollout import OK")
PY

python - <<'PY'
import os
import sys
from pathlib import Path
import yaml

project_root = Path(os.environ["PROJECT_ROOT"])
template_path = project_root / "vjepa2" / "configs" / "eval" / "vitg-384" / "ek100.yaml"
config_path = Path(os.environ["CONFIG_PATH"])

with template_path.open("r", encoding="utf-8") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

gaze_mode = os.environ["GAZE_MODE"]
if gaze_mode in {"pose_rnn_fuse", "multimodal_rnn_fuse", "binary_input_adapter_pose_rnn_fuse"}:
    os.environ["GAZE_RNN_USE_VIDEO_TOKEN"] = "0"

cfg["nodes"] = 1
cfg["tasks_per_node"] = int(os.environ["EVAL_GPUS"])
cfg["cpus_per_task"] = int(os.environ["EVAL_CPUS_PER_TASK"])
cfg["tag"] = os.environ["LORA_TAG"]
cfg["eval_name"] = "app.hdepic_lora_action_anticipation"
cfg["folder"] = os.environ["OUTPUT_DIR"]
cfg["resume_checkpoint"] = os.environ["RESUME_CHECKPOINT"] == "1"
cfg["val_only"] = os.environ.get("VAL_ONLY", "0") == "1"

data = cfg["experiment"]["data"]
data["dataset"] = "EK100"
data["file_format"] = 1
data["base_path"] = os.environ["ADAPTED_VIDEO_ROOT"]
data["dataset_train"] = str(Path(os.environ["ADAPTED_ANNOTATIONS_DIR"]) / "HD_EPIC_train_vjepa.csv")
data["dataset_val"] = str(Path(os.environ["ADAPTED_ANNOTATIONS_DIR"]) / "HD_EPIC_val_vjepa.csv")
data["dataset_test"] = str(Path(os.environ["ADAPTED_ANNOTATIONS_DIR"]) / "HD_EPIC_test_vjepa.csv")
data["num_workers"] = int(os.environ["EVAL_NUM_WORKERS"])
if os.environ.get("EVAL_VAL_NUM_WORKERS", "").strip():
    data["val_num_workers"] = int(os.environ["EVAL_VAL_NUM_WORKERS"])
data["clip_balanced"] = os.environ.get("EVAL_CLIP_BALANCED", "1") == "1"

cfg["experiment"]["optimization"]["batch_size"] = int(os.environ["EVAL_BATCH_SIZE"])
cfg["experiment"]["optimization"]["num_epochs"] = int(os.environ["EVAL_NUM_EPOCHS"])
if os.environ.get("EVAL_USE_BFLOAT16", "").strip():
    cfg["experiment"]["optimization"]["use_bfloat16"] = os.environ["EVAL_USE_BFLOAT16"].lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    print(f"[lora] Applied EVAL_USE_BFLOAT16={cfg['experiment']['optimization']['use_bfloat16']}")
if os.environ.get("EVAL_USE_FOCAL_LOSS", "").strip():
    cfg["experiment"]["optimization"]["use_focal_loss"] = os.environ["EVAL_USE_FOCAL_LOSS"].lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
multihead_kwargs = cfg["experiment"]["optimization"]["multihead_kwargs"]
if os.environ.get("EVAL_SINGLE_PROBE", "0").lower() in {"1", "true", "yes", "on"}:
    lr = float(os.environ.get("EVAL_LR") or "0.0001")
    start_lr = float(os.environ.get("EVAL_START_LR") or "0.0")
    final_lr = float(os.environ.get("EVAL_FINAL_LR") or "0.0")
    warmup = float(os.environ.get("EVAL_WARMUP_EPOCHS") or "2")
    weight_decay = float(os.environ.get("EVAL_WEIGHT_DECAY") or "0.0001")
    final_weight_decay = float(os.environ.get("EVAL_FINAL_WEIGHT_DECAY") or str(weight_decay))
    cfg["experiment"]["optimization"]["multihead_kwargs"] = [
        {
            "weight_decay": weight_decay,
            "final_weight_decay": final_weight_decay,
            "lr": lr,
            "start_lr": start_lr,
            "final_lr": final_lr,
            "warmup": warmup,
        }
    ]
    multihead_kwargs = cfg["experiment"]["optimization"]["multihead_kwargs"]
    print(
        "[lora] Using a single attention probe: "
        f"lr={lr:g} start_lr={start_lr:g} final_lr={final_lr:g} "
        f"warmup={warmup:g} wd={weight_decay:g} final_wd={final_weight_decay:g}"
    )
lr_scale_raw = os.environ.get("EVAL_LR_SCALE", "1").strip()
lr_scale = float(lr_scale_raw)
if lr_scale <= 0.0:
    raise SystemExit(f"Invalid EVAL_LR_SCALE={lr_scale_raw!r}; expected positive float")
if lr_scale != 1.0:
    scaled = 0
    for kwargs in multihead_kwargs:
        for key in ("lr", "start_lr", "ref_lr", "final_lr"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = float(kwargs[key]) * lr_scale
                scaled += 1
    print(f"[lora] Applied EVAL_LR_SCALE={lr_scale:g}; scaled {scaled} lr/start/ref/final lr values")
max_lr_raw = os.environ.get("EVAL_MAX_LR", "").strip()
if max_lr_raw:
    max_lr = float(max_lr_raw)
    if max_lr <= 0.0:
        raise SystemExit(f"Invalid EVAL_MAX_LR={max_lr_raw!r}; expected positive float")
    capped = 0
    for kwargs in multihead_kwargs:
        for key in ("lr", "start_lr", "ref_lr"):
            if key in kwargs and kwargs[key] is not None and float(kwargs[key]) > max_lr:
                kwargs[key] = max_lr
                capped += 1
    print(f"[lora] Applied EVAL_MAX_LR={max_lr:g} after scaling; capped {capped} lr/start/ref lr values")
for env_name, cfg_key in (
    ("EVAL_START_LR", "start_lr"),
    ("EVAL_FINAL_LR", "final_lr"),
    ("EVAL_WARMUP_EPOCHS", "warmup"),
):
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        continue
    value = float(raw)
    for kwargs in multihead_kwargs:
        kwargs[cfg_key] = value
    print(f"[lora] Applied {env_name}={value:g} to all {len(multihead_kwargs)} probe heads")
cfg["model_kwargs"]["checkpoint"] = os.environ["CHECKPOINT"]
hdepic_reference_overrides = {}
for env_name, _key in (
    ("HDEPIC_REFERENCE_ANNOTATIONS_PKL", "annotations_pkl"),
    ("HDEPIC_REFERENCE_VERB_CLASSES_CSV", "verb_classes_csv"),
    ("HDEPIC_REFERENCE_NOUN_CLASSES_CSV", "noun_classes_csv"),
):
    _val = os.environ.get(env_name, "").strip()
    if _val:
        hdepic_reference_overrides[_key] = _val
cfg["experiment"]["lora"] = {
    "enabled": True,
    "rank": int(os.environ["LORA_RANK"]),
    "alpha": float(os.environ["LORA_ALPHA"]),
    "dropout": float(os.environ["LORA_DROPOUT"]),
    "probe_train_mode": os.environ.get("LORA_PROBE_TRAIN_MODE", "lora"),
    "train_heads": os.environ.get("LORA_TRAIN_HEADS", "1").lower() in {"1", "true", "yes", "on"},
    "pretrained_probe": os.environ["LORA_PRETRAINED_PROBE"],
    "align_reference_metrics": True,
    "val_metric_scope": os.environ.get("LORA_VAL_METRIC_SCOPE", "native"),
    "output_regularization": {
        "enabled": float(os.environ.get("LORA_OUTPUT_REG_WEIGHT", "0") or "0") != 0.0,
        "mode": os.environ.get("LORA_OUTPUT_REG_MODE", "confidence_penalty"),
        "weight": float(os.environ.get("LORA_OUTPUT_REG_WEIGHT", "0") or "0"),
        "targets": [
            target.strip()
            for target in os.environ.get("LORA_OUTPUT_REG_TARGETS", "verb,noun,action").replace("|", ",").split(",")
            if target.strip()
        ],
    },
    "encoder_lora": {
        "enabled": os.environ.get("ENCODER_LORA_ENABLED", "0") == "1",
        "rank": int(os.environ.get("ENCODER_LORA_RANK", os.environ["LORA_RANK"])),
        "alpha": float(os.environ.get("ENCODER_LORA_ALPHA", os.environ["LORA_ALPHA"])),
        "dropout": float(os.environ.get("ENCODER_LORA_DROPOUT", os.environ["LORA_DROPOUT"])),
        "last_n_blocks": int(os.environ.get("ENCODER_LORA_LAST_N_BLOCKS", "12")),
        "lr_mult": float(os.environ.get("ENCODER_LORA_LR_MULT", "0.5")),
        "weight_decay": float(os.environ.get("ENCODER_LORA_WEIGHT_DECAY", "0.0001")),
        "activation_checkpointing": os.environ.get("ENCODER_LORA_ACTIVATION_CHECKPOINTING", "0") == "1",
    },
    "predictor_lora": {
        "enabled": os.environ.get("PREDICTOR_LORA_ENABLED", "0") == "1",
        "rank": int(os.environ.get("PREDICTOR_LORA_RANK", os.environ["LORA_RANK"])),
        "alpha": float(os.environ.get("PREDICTOR_LORA_ALPHA", os.environ["LORA_ALPHA"])),
        "dropout": float(os.environ.get("PREDICTOR_LORA_DROPOUT", os.environ["LORA_DROPOUT"])),
        "last_n_blocks": int(os.environ.get("PREDICTOR_LORA_LAST_N_BLOCKS", "0")),
        "lr_mult": float(os.environ.get("PREDICTOR_LORA_LR_MULT", "0.5")),
        "weight_decay": float(os.environ.get("PREDICTOR_LORA_WEIGHT_DECAY", "0.0001")),
        "activation_checkpointing": os.environ.get("PREDICTOR_LORA_ACTIVATION_CHECKPOINTING", "0") == "1",
    },
    "gaze": {
        "mode": os.environ["GAZE_MODE"],
        "gaze_root": os.environ["GAZE_ROOT"],
        "extract_root": os.environ["GAZE_EXTRACT_ROOT"],
        "sync_root": os.environ["GAZE_SYNC_ROOT"],
        "gamma": float(os.environ["GAZE_GATE_GAMMA"]),
        "sigma_px": float(os.environ["GAZE_GATE_SIGMA_PX"]),
        "learnable_gate": os.environ.get("GAZE_LEARNABLE_GATE", "1") == "1",
        "use_motion": os.environ["GAZE_MODE"] == "token_gate",
        "traj_len": int(os.environ.get("GAZE_RNN_TRAJ_LEN", "1024")),
        "rnn": {
            "hidden_dim": int(os.environ.get("GAZE_RNN_HIDDEN_DIM", "256")),
            "num_layers": int(os.environ.get("GAZE_RNN_NUM_LAYERS", "2")),
            "bidirectional": os.environ.get("GAZE_RNN_BIDIR", "1") == "1",
            "dropout": float(os.environ.get("GAZE_RNN_DROPOUT", "0.1")),
            "num_tokens": int(os.environ.get("GAZE_RNN_NUM_TOKENS", "64")),
            "use_video_tokens": os.environ.get("GAZE_RNN_USE_VIDEO_TOKEN", "0") == "1",
            "video_fusion": os.environ.get("GAZE_RNN_VIDEO_FUSION", "nearest_concat"),
            "video_proj_dim": int(os.environ.get("GAZE_RNN_VIDEO_PROJ_DIM", "128")),
            "local_temporal_radius": int(os.environ.get("GAZE_RNN_LOCAL_TEMPORAL_RADIUS", "0")),
            "local_spatial_radius": int(os.environ.get("GAZE_RNN_LOCAL_SPATIAL_RADIUS", "1")),
            "residual_alpha_init": float(os.environ.get("GAZE_RNN_RESIDUAL_ALPHA_INIT", "0.01")),
            "history_sec": float(os.environ.get("GAZE_RNN_HISTORY_SEC", "0")),
            "gaze_lr_mult": float(os.environ.get("GAZE_LR_MULT", "5.0")),
        },
        "hidden_dump": {
            "enabled": os.environ.get("GAZE_HIDDEN_DUMP", "0") == "1",
            "max_batches": 4,
        },
        "binary_radius_px": float(os.environ.get("GAZE_BINARY_RADIUS_PX", "64")),
        "binary_map_type": os.environ.get("GAZE_BINARY_MAP_TYPE", "binary"),
        "fallback_full_frame": os.environ.get("GAZE_BINARY_FALLBACK_FULL_FRAME", "0") == "1",
        "force_zero_map": os.environ.get("BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP", "0") == "1",
        "aug_aware": os.environ.get("BINARY_INPUT_ADAPTER_AUG_AWARE", "0") == "1",
        "disable_train_aug": os.environ.get(
            "BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG",
            "1"
            if os.environ["GAZE_MODE"]
            in {"binary_input_adapter", "binary_input_adapter_pose_rnn_fuse", "binary_input_adapter_gaze_pose_matrix"}
            else "0",
        )
        == "1",
        "input_adapter": {
            "hidden_dim": int(os.environ.get("BINARY_INPUT_ADAPTER_HIDDEN_DIM", "8")),
            "scale": float(os.environ.get("BINARY_INPUT_ADAPTER_SCALE", "1.0")),
            "temporal_kernel": int(
                os.environ.get(
                    "BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL",
                    "3" if os.environ["GAZE_MODE"] == "binary_input_adapter_gaze_pose_matrix" else "1",
                )
            ),
            "binary_center": float(os.environ.get("BINARY_INPUT_ADAPTER_BINARY_CENTER", "0.0")),
            "residual_clamp": float(os.environ.get("BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP", "1.0")),
            "lr_mult": float(os.environ.get("BINARY_INPUT_ADAPTER_LR_MULT", "0.05")),
            "weight_decay": float(os.environ.get("BINARY_INPUT_ADAPTER_WEIGHT_DECAY", "0.0001")),
            "activation_checkpointing": os.environ.get("BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING", "0") == "1",
            "in_channels": int(
                os.environ.get(
                    "BINARY_INPUT_ADAPTER_IN_CHANNELS",
                    "5" if os.environ["GAZE_MODE"] == "binary_input_adapter_gaze_pose_matrix" else "4",
                )
            ),
            **({"load_checkpoint_path": os.environ["BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT"]} if os.environ.get("BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT") else {}),
        },
        "pose_map": {
            "patch_height": int(os.environ.get("POSE_MAP_PATCH_HEIGHT", "128")),
            "patch_width": int(os.environ.get("POSE_MAP_PATCH_WIDTH", "9")),
            "layout": os.environ.get("POSE_MAP_LAYOUT", "topleft"),
            "normalize": os.environ.get("POSE_MAP_NORMALIZE", "none"),
            "force_zero_pose": os.environ.get("POSE_MAP_FORCE_ZERO", "0") == "1",
        },
        "encoder_output_adapter": {
            "num_heads": int(os.environ.get("ENCODER_GAZE_ADAPTER_NUM_HEADS", "4")),
            "dropout": float(os.environ.get("ENCODER_GAZE_ADAPTER_DROPOUT", "0.0")),
            "lr_mult": float(os.environ.get("ENCODER_GAZE_ADAPTER_LR_MULT", "0.05")),
            "weight_decay": float(os.environ.get("ENCODER_GAZE_ADAPTER_WEIGHT_DECAY", "0.0001")),
        },
        "pose": {
            "enabled": os.environ["GAZE_MODE"]
            in {
                "pose_rnn_fuse",
                "multimodal_rnn_fuse",
                "binary_input_adapter_pose_rnn_fuse",
                "binary_input_adapter_gaze_pose_matrix",
            },
            "slam_root": os.environ.get(
                "POSE_SLAM_ROOT",
                str(Path(os.environ["GAZE_ROOT"]) / "P01" / "SLAM" / "multi"),
            ),
            "mapping_json": os.environ.get(
                "POSE_MAPPING_JSON",
                str(Path(os.environ["GAZE_ROOT"]) / "P01" / "SLAM" / "multi" / "vrs_to_multi_slam.json"),
            ),
            "trajectory_file": os.environ.get("POSE_TRAJECTORY_FILE", "closed_loop_trajectory.csv"),
            "feature_set": os.environ.get("POSE_FEATURE_SET", "pose_6d"),
            "quality_min": float(os.environ.get("POSE_QUALITY_MIN", "0.0")),
            "history_sec": float(os.environ.get("POSE_HISTORY_SEC", os.environ.get("GAZE_RNN_HISTORY_SEC", "0"))),
            "interframe_k_max": int(os.environ.get("POSE_INTERFRAME_K_MAX", "128")),
        },
    },
    "prediction_dump": {
        "enabled": os.environ["PREDICTION_DUMP"] == "1",
        **({"path": os.environ["PREDICTION_DUMP_PATH"]} if os.environ.get("PREDICTION_DUMP_PATH") else {}),
    },
}
if hdepic_reference_overrides:
    cfg["experiment"]["lora"]["hdepic_reference"] = hdepic_reference_overrides
    print(f"[lora] Applied hdepic_reference overrides: {hdepic_reference_overrides}")

encoder_lora_targets = os.environ.get("ENCODER_LORA_TARGET_SUFFIXES", "").strip()
if encoder_lora_targets:
    cfg["experiment"]["lora"]["encoder_lora"]["target_suffixes"] = [
        suffix.strip()
        for suffix in encoder_lora_targets.replace("|", ",").split(",")
        if suffix.strip()
    ]

predictor_lora_targets = os.environ.get("PREDICTOR_LORA_TARGET_SUFFIXES", "").strip()
if predictor_lora_targets:
    cfg["experiment"]["lora"]["predictor_lora"]["target_suffixes"] = [
        suffix.strip()
        for suffix in predictor_lora_targets.replace("|", ",").split(",")
        if suffix.strip()
    ]

model_family = os.environ.get("MODEL_FAMILY", "vjepa").strip().lower()
backbone = os.environ.get("BACKBONE", "vitg").strip().lower()
if model_family == "vlm":
    model_id = os.environ.get("VLM_MODEL_ID", "").strip() or os.environ.get("CHECKPOINT", "").strip()
    if not model_id:
        raise SystemExit("MODEL_FAMILY=vlm requires VLM_MODEL_ID or CHECKPOINT")
    resolution = int(os.environ.get("EVAL_RESOLUTION", os.environ.get("VLM_IMAGE_SIZE", "224")))
    # Decode only the frames consumed by the frozen VLM while preserving the
    # template's real-time observation window.  The old path decoded 32 frames
    # at 8 fps, then selected 8 frames on the GPU; 24/32 decoded frames were
    # discarded after paying the full CPU/video-I/O cost.
    source_frames = int(data["frames_per_clip"])
    source_fps = float(data["frames_per_second"])
    if source_frames <= 0 or source_fps <= 0:
        raise SystemExit(
            f"Invalid source video sampling: frames_per_clip={source_frames} frames_per_second={source_fps}"
        )
    observation_sec = source_frames / source_fps
    vlm_num_frames = int(os.environ.get("VLM_NUM_FRAMES", "8"))
    if vlm_num_frames <= 0:
        raise SystemExit(f"VLM_NUM_FRAMES must be positive, got {vlm_num_frames}")
    data["frames_per_clip"] = vlm_num_frames
    data["frames_per_second"] = vlm_num_frames / observation_sec
    data["resolution"] = resolution
    data["anticipation_time_sec"] = [1.0, 1.0]
    data["train_anticipation_point"] = [1.0, 1.0]
    data["train_anticipation_time_sec"] = [1.0, 1.0]
    cfg["model_kwargs"]["checkpoint"] = model_id
    cfg["model_kwargs"]["module_name"] = "app.hdepic_lora_action_anticipation.modelcustom.vlm_video_encoder"
    cfg["model_kwargs"]["pretrain_kwargs"] = {
        "model_id": model_id,
        "model_class": os.environ.get("VLM_MODEL_CLASS", "auto"),
        "num_frames": vlm_num_frames,
        "token_mode": os.environ.get("VLM_TOKEN_MODE", "pooled"),
        "image_size": int(os.environ.get("VLM_IMAGE_SIZE", str(resolution))),
        "trust_remote_code": os.environ.get("VLM_TRUST_REMOTE_CODE", "0").lower() in {"1", "true", "yes", "on"},
        "local_files_only": os.environ.get("VLM_LOCAL_FILES_ONLY", "0").lower() in {"1", "true", "yes", "on"},
        "gpu_pulse_iters": int(os.environ.get("VLM_GPU_PULSE_ITERS", "0") or "0"),
        "gpu_pulse_size": int(os.environ.get("VLM_GPU_PULSE_SIZE", "512") or "512"),
        "mllama_chunk_size": int(os.environ.get("VLM_MLLAMA_CHUNK_SIZE", "8") or "8"),
    }
    if os.environ.get("VLM_TORCH_DTYPE", "").strip():
        cfg["model_kwargs"]["pretrain_kwargs"]["torch_dtype"] = os.environ["VLM_TORCH_DTYPE"].strip()
    cfg["model_kwargs"]["wrapper_kwargs"] = {}
    cfg["experiment"]["lora"]["pretrained_probe"] = ""
    cfg["experiment"]["lora"]["encoder_lora"]["enabled"] = False
    cfg["experiment"]["lora"]["predictor_lora"]["enabled"] = False
    print(
        "[lora] Applied MODEL_FAMILY=vlm: "
        f"model_id={model_id} class={cfg['model_kwargs']['pretrain_kwargs']['model_class']} "
        f"num_frames={cfg['model_kwargs']['pretrain_kwargs']['num_frames']} "
        f"data_sampling={data['frames_per_clip']}@{data['frames_per_second']:g}fps "
        f"observation={observation_sec:g}s "
        f"token_mode={cfg['model_kwargs']['pretrain_kwargs']['token_mode']} "
        f"resolution={resolution} "
        f"gpu_pulse_iters={cfg['model_kwargs']['pretrain_kwargs']['gpu_pulse_iters']}; "
        "pretrained_probe disabled"
    )
elif backbone in {"vitl", "vit-l", "vit_large"}:
    vitl_template = project_root / "vjepa2" / "configs" / "eval" / "vitl" / "ek100.yaml"
    with vitl_template.open("r", encoding="utf-8") as f:
        vitl_cfg = yaml.load(f, Loader=yaml.FullLoader)
    vitl_model_kwargs = vitl_cfg["model_kwargs"]
    cfg["model_kwargs"]["checkpoint"] = os.environ.get(
        "CHECKPOINT",
        str(project_root / "checkpoints" / "vitl.pt"),
    )
    cfg["model_kwargs"]["pretrain_kwargs"] = vitl_model_kwargs["pretrain_kwargs"]
    resolution = int(os.environ.get("EVAL_RESOLUTION", "256"))
    data["resolution"] = resolution
    enc_name = vitl_model_kwargs["pretrain_kwargs"]["encoder"]["model_name"]
    print(
        f"[lora] Applied BACKBONE=vitl: encoder={enc_name} resolution={resolution} "
        f"checkpoint={cfg['model_kwargs']['checkpoint']}"
    )

def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}

_ar10s_triggers = []
if _truthy_env("AR10S_SLIDING_PROBE"):
    _ar10s_triggers.append("AR10S_SLIDING_PROBE=1")
if _truthy_env("HDEPIC_AR10S_SLIDING_GAZE_POSE"):
    _ar10s_triggers.append("HDEPIC_AR10S_SLIDING_GAZE_POSE=1")
_antic_raw = os.environ.get("EVAL_ANTICIPATION_SEC", "").strip()
if _antic_raw:
    _ar10s_triggers.append(f"EVAL_ANTICIPATION_SEC={_antic_raw!r}")

ar10s_sliding = model_family != "vlm" and not _truthy_env("HDEPIC_DIRECT_ROPE_10S") and bool(_ar10s_triggers)
direct_rope_10s = model_family != "vlm" and _truthy_env("HDEPIC_DIRECT_ROPE_10S")

if direct_rope_10s:
    val_antic = float(os.environ.get("EVAL_ANTICIPATION_SEC", "10"))
    train_min_raw = os.environ.get("EVAL_TRAIN_ANTICIPATION_SEC_MIN", "").strip()
    train_max_raw = os.environ.get("EVAL_TRAIN_ANTICIPATION_SEC_MAX", "").strip()
    train_min = float(train_min_raw) if train_min_raw else max(val_antic - 2.0, 0.0)
    train_max = float(train_max_raw) if train_max_raw else val_antic
    if train_min > train_max:
        raise SystemExit(f"Invalid train anticipation range [{train_min}, {train_max}]")
    rope_mode = os.environ.get("MODEL_ROPE_SCALE_MODE", "ntk_temporal").strip() or "ntk_temporal"

    data["anticipation_time_sec"] = [val_antic, val_antic]
    data["train_anticipation_point"] = [0.0, 0.0]
    data["train_anticipation_time_sec"] = [train_min, train_max]

    cfg["model_kwargs"]["module_name"] = (
        "app.hdepic_lora_action_anticipation.modelcustom.vit_encoder_predictor_direct_rope"
    )
    wrapper = cfg["model_kwargs"].setdefault("wrapper_kwargs", {})
    wrapper["no_predictor"] = False
    wrapper.setdefault("num_output_frames", 2)
    wrapper.setdefault("num_steps", 1)
    wrapper["rope_scale_mode"] = rope_mode

    print(
        "[lora] Applied direct_rope 10s config: "
        f"val_anticipation={val_antic}s train_anticipation=[{train_min}, {train_max}] "
        f"rope_scale_mode={rope_mode} clip_balanced={data.get('clip_balanced')}"
    )
elif ar10s_sliding:
    val_antic = float(os.environ.get("EVAL_ANTICIPATION_SEC", "10"))
    train_min_raw = os.environ.get("EVAL_TRAIN_ANTICIPATION_SEC_MIN", "").strip()
    train_max_raw = os.environ.get("EVAL_TRAIN_ANTICIPATION_SEC_MAX", "").strip()
    train_min = float(train_min_raw) if train_min_raw else max(val_antic - 2.0, 0.0)
    train_max = float(train_max_raw) if train_max_raw else val_antic
    if train_min > train_max:
        raise SystemExit(f"Invalid train anticipation range [{train_min}, {train_max}]")
    return_mode = os.environ.get("MODEL_RETURN_MODE", "final_window").strip() or "final_window"
    max_rollout = int(os.environ.get("MODEL_MAX_ROLLOUT_STEPS", "512"))

    data["anticipation_time_sec"] = [val_antic, val_antic]
    data["train_anticipation_point"] = [0.0, 0.0]
    data["train_anticipation_time_sec"] = [train_min, train_max]

    cfg["model_kwargs"]["module_name"] = (
        "app.hdepic_lora_action_anticipation.modelcustom.vit_encoder_predictor_rollout"
    )
    wrapper = cfg["model_kwargs"].setdefault("wrapper_kwargs", {})
    wrapper["no_predictor"] = False
    wrapper.setdefault("num_output_frames", 2)
    ar_steps_raw = os.environ.get("HDEPIC_AR_STEPS", os.environ.get("MODEL_NUM_STEPS", "")).strip()
    wrapper["num_steps"] = int(ar_steps_raw) if ar_steps_raw else int(wrapper.get("num_steps", 1))
    wrapper["return_mode"] = return_mode
    wrapper["max_rollout_steps"] = max_rollout

    print(
        "[lora] Applied AR10s sliding config: "
        f"val_anticipation={val_antic}s train_anticipation=[{train_min}, {train_max}] "
        f"return_mode={return_mode} num_steps={wrapper['num_steps']} "
        f"max_rollout_steps={max_rollout} "
        f"clip_balanced={data.get('clip_balanced')}"
    )
    print(f"[lora] AR10s mode triggered by: {', '.join(_ar10s_triggers)}")
elif model_family == "vlm":
    print(
        "[lora] Mode: 1s VLM observed-clip baseline; "
        f"anticipation_time_sec={data.get('anticipation_time_sec')} "
        f"module_name={cfg['model_kwargs'].get('module_name')}"
    )
elif not direct_rope_10s:
    print(
        "[lora] Mode: 1s concat_ar (default); "
        f"anticipation_time_sec={data.get('anticipation_time_sec')} "
        f"module_name={cfg['model_kwargs'].get('module_name')}"
    )

val_every_raw = os.environ.get("EVAL_VAL_EVERY", "").strip()
if val_every_raw:
    cfg["experiment"]["optimization"]["val_every_epochs"] = int(val_every_raw)
    print(f"[lora] Applied EVAL_VAL_EVERY={cfg['experiment']['optimization']['val_every_epochs']}")
cfg["experiment"]["optimization"]["best_metric"] = os.environ.get("EVAL_BEST_METRIC", "val-action-acc")
cfg["experiment"]["optimization"]["early_stopping_patience"] = int(
    os.environ.get("EVAL_EARLY_STOPPING_PATIENCE", "0") or "0"
)
print(
    "[lora] Checkpoint policy: "
    f"best_metric={cfg['experiment']['optimization']['best_metric']} "
    f"early_stopping_patience={cfg['experiment']['optimization']['early_stopping_patience']}"
)

config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"[lora] Wrote config: {config_path}")

if direct_rope_10s:
    mk = cfg["model_kwargs"]
    wrapper = mk.get("wrapper_kwargs", {})
    print("[direct-rope-10s] config check:")
    print(f"  module_name={mk.get('module_name')}")
    print(f"  rope_scale_mode={wrapper.get('rope_scale_mode')} (single predictor fwd, NOT rollout)")
    print(f"  train_anticipation_time_sec={data.get('train_anticipation_time_sec')}")
    print(f"  val_anticipation_time_sec={data.get('anticipation_time_sec')}")
    print(f"  gaze.mode={cfg['experiment']['lora'].get('gaze', {}).get('mode')}")
    print(
        "  encoder_lora.enabled="
        f"{cfg['experiment']['lora'].get('encoder_lora', {}).get('enabled')}"
    )
    errors = []
    if "vit_encoder_predictor_direct_rope" not in str(mk.get("module_name", "")):
        errors.append("module_name must be vit_encoder_predictor_direct_rope")
    val_antic = float(os.environ.get("EVAL_ANTICIPATION_SEC", "10"))
    if data.get("anticipation_time_sec") != [val_antic, val_antic]:
        errors.append(f"val anticipation must be [{val_antic}, {val_antic}]")
    if wrapper.get("rope_scale_mode") != (
        os.environ.get("MODEL_ROPE_SCALE_MODE", "ntk_temporal").strip() or "ntk_temporal"
    ):
        errors.append("rope_scale_mode mismatch")
    if errors:
        for err in errors:
            print(f"[direct-rope-10s] FATAL: {err}", file=sys.stderr)
        raise SystemExit(1)
elif (
    os.environ.get("AR10S_SLIDING_PROBE", "0").lower() in {"1", "true", "yes", "on"}
    or os.environ.get("HDEPIC_AR10S_SLIDING_GAZE_POSE", "0").lower() in {"1", "true", "yes", "on"}
    or (
        os.environ.get("EVAL_ANTICIPATION_SEC", "").strip()
        and not _truthy_env("HDEPIC_DIRECT_ROPE_10S")
    )
):
    mk = cfg["model_kwargs"]
    wrapper = mk.get("wrapper_kwargs", {})
    print("[ar10s-sliding] config check:")
    print(f"  module_name={mk.get('module_name')}")
    print(
        "  return_mode="
        f"{wrapper.get('return_mode')} (final_window = JEPA_ARVR slid window, NOT concat_ar)"
    )
    print(f"  max_rollout_steps={wrapper.get('max_rollout_steps')}")
    print(f"  num_steps={wrapper.get('num_steps')} (>1 = JEPA_ARVR coarse sliding, ~3-4s/step)")
    print(f"  train_anticipation_time_sec={data.get('train_anticipation_time_sec')}")
    print(f"  val_anticipation_time_sec={data.get('anticipation_time_sec')}")
    print(f"  gaze.mode={cfg['experiment']['lora'].get('gaze', {}).get('mode')}")
    print(
        "  encoder_lora.enabled="
        f"{cfg['experiment']['lora'].get('encoder_lora', {}).get('enabled')}"
    )
    errors = []
    if "vit_encoder_predictor_rollout" not in str(mk.get("module_name", "")):
        errors.append("module_name must be vit_encoder_predictor_rollout")
    val_antic = float(os.environ.get("EVAL_ANTICIPATION_SEC", "10"))
    if data.get("anticipation_time_sec") != [val_antic, val_antic]:
        errors.append(f"val anticipation must be [{val_antic}, {val_antic}]")
    if wrapper.get("return_mode") != (os.environ.get("MODEL_RETURN_MODE", "final_window").strip() or "final_window"):
        errors.append("return_mode mismatch")
    if errors:
        for err in errors:
            print(f"[ar10s-sliding] FATAL: {err}", file=sys.stderr)
        raise SystemExit(1)
PY

cd "${VJEPA_ROOT}"
device_args=()
for ((gpu_idx=0; gpu_idx<${EVAL_GPUS}; gpu_idx++)); do
    device_args+=("cuda:${gpu_idx}")
done
debugmode="${EVAL_DEBUGMODE}"
if [[ "${debugmode}" == "auto" || -z "${debugmode}" ]]; then
    if [[ "${EVAL_GPUS}" -le 1 ]]; then
        debugmode=1
    else
        debugmode=0
    fi
fi
eval_cmd=(python -m evals.main --fname "${CONFIG_PATH}" --devices "${device_args[@]}")
if [[ "${debugmode}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    eval_cmd+=(--debugmode true)
fi
echo "[lora] Launch command: ${eval_cmd[*]}"
"${eval_cmd[@]}"
BASH

monitor_pids=()
stop_perf_monitors() {
    for pid in "${monitor_pids[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}

sample_cgroup_memory() {
    local out_path="$1"
    local interval="$2"
    local err_path="${out_path%.csv}.err"
    local cg_rel=""
    local cg_base=""
    local mem_current_file=""
    local mem_peak_file=""
    local mem_max_file=""
    local current_bytes=""
    local peak_bytes=""
    local limit_bytes=""
    local current_mib=""
    local peak_mib=""
    local limit_mib=""

    if [[ -r /proc/self/cgroup ]]; then
        cg_rel="$(awk -F: '$2 == "" {print $3; exit}' /proc/self/cgroup || true)"
    fi

    for cg_base in "/sys/fs/cgroup${cg_rel}" "/sys/fs/cgroup"; do
        if [[ -f "${cg_base}/memory.current" ]]; then
            mem_current_file="${cg_base}/memory.current"
            [[ -f "${cg_base}/memory.peak" ]] && mem_peak_file="${cg_base}/memory.peak"
            [[ -f "${cg_base}/memory.max" ]] && mem_max_file="${cg_base}/memory.max"
            break
        fi
        if [[ -f "${cg_base}/memory/memory.usage_in_bytes" ]]; then
            mem_current_file="${cg_base}/memory/memory.usage_in_bytes"
            [[ -f "${cg_base}/memory/memory.max_usage_in_bytes" ]] && mem_peak_file="${cg_base}/memory/memory.max_usage_in_bytes"
            [[ -f "${cg_base}/memory/memory.limit_in_bytes" ]] && mem_max_file="${cg_base}/memory/memory.limit_in_bytes"
            break
        fi
    done

    if [[ -z "${mem_current_file}" ]]; then
        {
            echo "timestamp,cgroup_path,current_mib,peak_mib,limit_mib,slurm_mem_per_node_mb"
            echo "$(date '+%Y-%m-%dT%H:%M:%S%z'),not_found,,,,${SLURM_MEM_PER_NODE:-}"
        } > "${out_path}" 2> "${err_path}"
        return 0
    fi

    # Subshell redirect (not "{ ...; } > file") — avoids bash issues with return/local inside
    # redirected compound commands on some cluster /bin/bash builds (Slurm exit 2 after train).
    (
        echo "timestamp,cgroup_path,current_mib,peak_mib,limit_mib,slurm_mem_per_node_mb"
        while true; do
            current_bytes=""
            peak_bytes=""
            limit_bytes=""
            current_mib=""
            peak_mib=""
            limit_mib=""
            current_bytes="$(cat "${mem_current_file}" 2>/dev/null || true)"
            [[ -n "${mem_peak_file}" ]] && peak_bytes="$(cat "${mem_peak_file}" 2>/dev/null || true)"
            [[ -n "${mem_max_file}" ]] && limit_bytes="$(cat "${mem_max_file}" 2>/dev/null || true)"
            if [[ "${current_bytes}" =~ ^[0-9]+$ ]]; then
                current_mib=$(( current_bytes / 1024 / 1024 ))
            fi
            if [[ "${peak_bytes}" =~ ^[0-9]+$ ]]; then
                peak_mib=$(( peak_bytes / 1024 / 1024 ))
            fi
            if [[ "${limit_bytes}" =~ ^[0-9]+$ ]]; then
                limit_mib=$(( limit_bytes / 1024 / 1024 ))
            elif [[ "${limit_bytes}" == "max" ]]; then
                limit_mib="max"
            fi
            echo "$(date '+%Y-%m-%dT%H:%M:%S%z'),${cg_rel:-/},${current_mib},${peak_mib},${limit_mib},${SLURM_MEM_PER_NODE:-}"
            sleep "${interval}"
        done
    ) > "${out_path}" 2> "${err_path}"
}

if [[ "${PERF_MONITOR}" == "1" ]]; then
    perf_prefix="${PROJECT_ROOT}/logs/perf_${SLURM_JOB_ID:-manual}"
    echo "[perf] Sampling nvidia-smi/vmstat/cgroup-memory every ${PERF_INTERVAL}s into ${perf_prefix}.*"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
            --format=csv -l "${PERF_INTERVAL}" > "${perf_prefix}_nvidia_smi.csv" 2> "${perf_prefix}_nvidia_smi.err" &
        monitor_pids+=("$!")
    else
        echo "[perf] nvidia-smi not available on host" > "${perf_prefix}_nvidia_smi.err"
    fi
    if command -v vmstat >/dev/null 2>&1; then
        vmstat -t "${PERF_INTERVAL}" > "${perf_prefix}_vmstat.log" 2> "${perf_prefix}_vmstat.err" &
        monitor_pids+=("$!")
    else
        echo "[perf] vmstat not available on host" > "${perf_prefix}_vmstat.err"
    fi
    sample_cgroup_memory "${perf_prefix}_cgroup_memory.csv" "${PERF_INTERVAL}" &
    monitor_pids+=("$!")
    trap stop_perf_monitors EXIT
fi

train_rc=0
if singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"; then
    train_rc=0
else
    train_rc=$?
fi

trap - EXIT
stop_perf_monitors
exit "${train_rc}"
