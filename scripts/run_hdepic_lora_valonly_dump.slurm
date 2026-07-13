#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_valonly_dump
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240GB
#SBATCH --time=01:59:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_valonly_dump_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_valonly_dump_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Val-only run with PredictionDumper enabled, for an already-trained LoRA ckpt.
# Reuses an existing val-only yaml under configs/generated/ (one per target
# ckpt) and injects experiment.lora.prediction_dump.enabled=true so each
# validation produces a per-sample CSV under outputs/.../<tag>/val_predictions.csv.
#
# Usage:
#   sbatch --export=ALL,SOURCE_YAML=configs/generated/future_latent_compare/path_y_b1_clean_10s.yaml,VAL_TAG=valdump-b1 scripts/run_hdepic_lora_valonly_dump.slurm
#   sbatch --export=ALL,SOURCE_YAML=configs/generated/future_latent_compare/path_y_b2_rnn_gaze_10s.yaml,VAL_TAG=valdump-b2 scripts/run_hdepic_lora_valonly_dump.slurm
#   sbatch --export=ALL,SOURCE_YAML=configs/generated/future_latent_compare/path_y_b5_binary_adapter_10s.yaml,VAL_TAG=valdump-b5 scripts/run_hdepic_lora_valonly_dump.slurm
#
# Optional overrides:
#   ANTICIPATION_SEC=1.0   override anticipation_time_sec for both ends
#   EVAL_BATCH_SIZE=1
#   EVAL_VAL_NUM_WORKERS=2
#   LORA_VAL_METRIC_SCOPE=native
#   LORA_VAL_METRIC_AGGREGATION=single_head
#   VAL_FIXED_HEAD_INDEX=3   # optional; omit to auto-pick head by action Top-3 under scope
#   MODEL_MODULE_NAME=...  override the model wrapper, e.g. AR rollout at 10s
#   MODEL_MAX_ROLLOUT_STEPS=512
#   BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT=/abs/path/binary_input_adapter_latest.pt

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

CONFIG_DIR="${CONFIG_DIR:-${PROJECT_ROOT}/configs/generated/valonly_dump}"
SOURCE_YAML="${SOURCE_YAML:?must set SOURCE_YAML to an existing val-only yaml}"
VAL_TAG="${VAL_TAG:?must set VAL_TAG (short label for the generated config file)}"
ANTICIPATION_SEC="${ANTICIPATION_SEC:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-2}"
DUMP_PATH="${DUMP_PATH:-}"
LORA_VAL_METRIC_SCOPE="${LORA_VAL_METRIC_SCOPE:-native}"
LORA_VAL_METRIC_AGGREGATION="${LORA_VAL_METRIC_AGGREGATION:-metric_wise_max}"
VAL_FIXED_HEAD_INDEX="${VAL_FIXED_HEAD_INDEX:-}"
MODEL_MODULE_NAME="${MODEL_MODULE_NAME:-}"
MODEL_MAX_ROLLOUT_STEPS="${MODEL_MAX_ROLLOUT_STEPS:-}"
BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT="${BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT:-}"
BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP="${BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP:-0}"
POSE_MAP_FORCE_ZERO="${POSE_MAP_FORCE_ZERO:-0}"
VALDUMP_PREDICTION_DUMP="${VALDUMP_PREDICTION_DUMP:-1}"

mkdir -p "${PROJECT_ROOT}/logs" "${CONFIG_DIR}"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export SOURCE_YAML VAL_TAG CONFIG_DIR ANTICIPATION_SEC EVAL_BATCH_SIZE EVAL_VAL_NUM_WORKERS DUMP_PATH
export LORA_VAL_METRIC_SCOPE LORA_VAL_METRIC_AGGREGATION VAL_FIXED_HEAD_INDEX
export MODEL_MODULE_NAME MODEL_MAX_ROLLOUT_STEPS BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT
export BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP POSE_MAP_FORCE_ZERO VALDUMP_PREDICTION_DUMP
export LORA_VAL_METRIC_AGGREGATION VAL_FIXED_HEAD_INDEX

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

src = Path(os.environ["SOURCE_YAML"])
if not src.is_absolute():
    src = Path(os.environ["PROJECT_ROOT"]) / src
if not src.exists():
    raise SystemExit(f"[valdump] SOURCE_YAML not found: {src}")

with src.open("r", encoding="utf-8") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg["val_only"] = True
cfg["resume_checkpoint"] = True
cfg["experiment"]["optimization"]["batch_size"] = int(os.environ["EVAL_BATCH_SIZE"])
cfg["experiment"]["optimization"]["num_epochs"] = 1
data = cfg["experiment"]["data"]
data["num_workers"] = max(int(data.get("num_workers", 2)), 1)
data["val_num_workers"] = int(os.environ["EVAL_VAL_NUM_WORKERS"])
antic = os.environ.get("ANTICIPATION_SEC", "").strip()
if antic:
    a = float(antic)
    data["anticipation_time_sec"] = [a, a]

lora = cfg["experiment"]["lora"]
lora["val_metric_scope"] = os.environ.get("LORA_VAL_METRIC_SCOPE", "native")
agg = os.environ.get("LORA_VAL_METRIC_AGGREGATION", "metric_wise_max")
if agg == "action_top3_single_head":
    agg = "single_head"
lora["val_metric_aggregation"] = agg
fixed_head = os.environ.get("VAL_FIXED_HEAD_INDEX", "").strip()
if fixed_head:
    lora["val_fixed_head_index"] = int(fixed_head)
elif "val_fixed_head_index" in lora:
    lora.pop("val_fixed_head_index", None)
module_name = os.environ.get("MODEL_MODULE_NAME", "").strip()
if module_name:
    cfg["model_kwargs"]["module_name"] = module_name
    if "vit_encoder_predictor_rollout" not in module_name:
        wrapper = cfg["model_kwargs"].setdefault("wrapper_kwargs", {})
        for key in ("max_rollout_steps", "rollout_stride_chunks", "return_mode"):
            wrapper.pop(key, None)
max_rollout_steps = os.environ.get("MODEL_MAX_ROLLOUT_STEPS", "").strip()
if max_rollout_steps:
    cfg["model_kwargs"].setdefault("wrapper_kwargs", {})["max_rollout_steps"] = int(max_rollout_steps)

gaze = dict(lora.get("gaze", {}))
gaze_mode = str(gaze.get("mode", "")).lower()
if gaze_mode.startswith("binary_input_adapter"):
    input_adapter = dict(gaze.get("input_adapter", {}))
    adapter_ckpt = os.environ.get("BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT", "").strip()
    if not adapter_ckpt:
        adapter_ckpt = str(
            Path(cfg["folder"])
            / "action_anticipation_frozen"
            / cfg["tag"]
            / "binary_input_adapter_latest.pt"
        )
    input_adapter["load_checkpoint_path"] = adapter_ckpt
    gaze["input_adapter"] = input_adapter
    lora["gaze"] = gaze

if os.environ.get("BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP", "").strip().lower() in {"1", "true", "yes", "on"}:
    gaze["force_zero_map"] = True
if os.environ.get("POSE_MAP_FORCE_ZERO", "").strip().lower() in {"1", "true", "yes", "on"}:
    gaze.setdefault("pose_map", {})["force_zero_pose"] = True
lora["gaze"] = gaze

pd_cfg = dict(lora.get("prediction_dump", {}))
pd_cfg["enabled"] = os.environ.get("VALDUMP_PREDICTION_DUMP", "1").strip().lower() not in {"0", "false", "no", "off"}
pd_cfg.setdefault("topk", 10)
dump_path = os.environ.get("DUMP_PATH", "").strip()
if dump_path:
    pd_cfg["path"] = dump_path
lora["prediction_dump"] = pd_cfg

# Past-window block, if present, must not silently apply to train.
past = lora.get("past_window_baseline")
if isinstance(past, dict) and past.get("enabled", False):
    past.setdefault("apply_to_train", False)

config_dir = Path(os.environ["CONFIG_DIR"])
config_dir.mkdir(parents=True, exist_ok=True)
out_path = config_dir / f"{os.environ['VAL_TAG']}.yaml"
with out_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[valdump] Wrote {out_path}")
print(
    f"[valdump] tag={cfg.get('tag')} dump_enabled=True "
    f"anticipation_time_sec={data['anticipation_time_sec']} "
    f"val_metric_scope={lora['val_metric_scope']} "
    f"val_metric_aggregation={lora.get('val_metric_aggregation', 'metric_wise_max')} "
    f"val_fixed_head_index={lora.get('val_fixed_head_index', '')} "
    f"model_module={cfg['model_kwargs']['module_name']}"
)
if gaze_mode.startswith("binary_input_adapter"):
    print(f"[valdump] gaze_mode={gaze_mode} binary_input_adapter_load_checkpoint={gaze['input_adapter']['load_checkpoint_path']}")
PY

config_path="${CONFIG_DIR}/${VAL_TAG}.yaml"
cd "${VJEPA_ROOT}"
echo "[valdump] launching ${config_path}"
python -m evals.main \
    --fname "${config_path}" \
    --devices cuda:0 \
    --debugmode true
BASH
)

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
