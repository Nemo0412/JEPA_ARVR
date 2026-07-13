#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_future_latents
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=18:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_future_latents_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_future_latents_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Compare latent sources on the same HD-EPIC validation samples:
#   encoder: observed clip encoder tokens only, no predictor call
#   direct_single: one mask span placed at the requested future target chunk
#   direct_dense: one larger mask span covering intermediate chunks through target
#   ar: predictor rollout one valid local chunk at a time
#   oracle: encoder target chunk from the actual future/action-time clip
#
# Outputs one CSV with latent distance-to-oracle and action/verb/noun metrics.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
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
SOURCE_LORA_TAG="${SOURCE_LORA_TAG:-${LORA_SOURCE_TAG:-hdepic-lora-vitg16-384-cleanp01-refmetrics}}"

HORIZONS="${HORIZONS:-1 1.5 2 2.5 3 4 5 6 7 8 9 10 60}"
VIDEO_DOWNSAMPLE_FACTOR="${VIDEO_DOWNSAMPLE_FACTOR:-1}"
DOWNSAMPLE_TAG="${VIDEO_DOWNSAMPLE_FACTOR}"
if [[ "${DOWNSAMPLE_TAG}" == *.0 ]]; then
    DOWNSAMPLE_TAG="${DOWNSAMPLE_TAG%.0}"
else
    DOWNSAMPLE_TAG="${DOWNSAMPLE_TAG//./p}"
fi
LORA_TAG="${LORA_TAG:-${SOURCE_LORA_TAG}-futurelatents-ds${DOWNSAMPLE_TAG}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DROP_INCOMPLETE_HISTORY="${DROP_INCOMPLETE_HISTORY:-1}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MAX_ROLLOUT_STEPS="${MAX_ROLLOUT_STEPS:-512}"
METRIC_SCOPE="${METRIC_SCOPE:-native}"
HEAD_SELECTION="${HEAD_SELECTION:-vjepa2}"
RESULT_CSV="${RESULT_CSV:-${OUTPUT_DIR}/future_latent_compare/${LORA_TAG}/future_latent_compare.csv}"
CONFIG_PATH="${CONFIG_PATH:-${CONFIG_DIR}/hdepic_future_latent_compare.yaml}"

GAZE_MODE="${GAZE_MODE:-none}"
GAZE_ROOT="${GAZE_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/SLAM-and-Gaze}"
GAZE_EXTRACT_ROOT="${GAZE_EXTRACT_ROOT:-${PROJECT_ROOT}/outputs/hdepic_gaze_token_gate/extracted}"
GAZE_SYNC_ROOT="${GAZE_SYNC_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/Videos}"
GAZE_BINARY_RADIUS_PX="${GAZE_BINARY_RADIUS_PX:-64}"
BINARY_INPUT_ADAPTER_HIDDEN_DIM="${BINARY_INPUT_ADAPTER_HIDDEN_DIM:-8}"
BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL="${BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL:-1}"
BINARY_INPUT_ADAPTER_SCALE="${BINARY_INPUT_ADAPTER_SCALE:-1.0}"
BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP="${BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP:-1.0}"
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

mkdir -p "${PROJECT_ROOT}/logs" "${CONFIG_DIR}" "$(dirname "${RESULT_CSV}")"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export ADAPTED_ANNOTATIONS_DIR ADAPTED_VIDEO_ROOT CONFIG_DIR CONFIG_PATH CHECKPOINT OUTPUT_DIR
export LORA_PRETRAINED_PROBE SOURCE_LORA_TAG LORA_TAG HORIZONS VIDEO_DOWNSAMPLE_FACTOR EVAL_BATCH_SIZE EVAL_NUM_WORKERS MAX_SAMPLES
export DROP_INCOMPLETE_HISTORY LORA_RANK LORA_ALPHA LORA_DROPOUT MAX_ROLLOUT_STEPS METRIC_SCOPE HEAD_SELECTION RESULT_CSV
export GAZE_MODE GAZE_ROOT GAZE_EXTRACT_ROOT GAZE_SYNC_ROOT GAZE_BINARY_RADIUS_PX
export BINARY_INPUT_ADAPTER_HIDDEN_DIM BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL BINARY_INPUT_ADAPTER_SCALE BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP
export GAZE_RNN_HIDDEN_DIM GAZE_RNN_NUM_LAYERS GAZE_RNN_NUM_TOKENS GAZE_RNN_DROPOUT GAZE_RNN_BIDIR GAZE_RNN_TRAJ_LEN
export GAZE_RNN_USE_VIDEO_TOKEN GAZE_RNN_VIDEO_FUSION GAZE_RNN_VIDEO_PROJ_DIM GAZE_RNN_LOCAL_TEMPORAL_RADIUS GAZE_RNN_LOCAL_SPATIAL_RADIUS GAZE_RNN_RESIDUAL_ALPHA_INIT GAZE_RNN_HISTORY_SEC

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
import shutil
import yaml

project_root = Path(os.environ["PROJECT_ROOT"])
template_path = project_root / "vjepa2" / "configs" / "eval" / "vitg-384" / "ek100.yaml"
with template_path.open("r", encoding="utf-8") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

output_dir = Path(os.environ["OUTPUT_DIR"])
source_tag = os.environ["SOURCE_LORA_TAG"].strip()
run_tag = os.environ["LORA_TAG"].strip()
if run_tag == source_tag:
    raise SystemExit(
        "Refusing to use SOURCE_LORA_TAG as output LORA_TAG; "
        f"set LORA_TAG to a new experiment namespace (source={source_tag})"
    )
source_latest = output_dir / "action_anticipation_frozen" / source_tag / "latest.pt"
if not source_latest.exists():
    raise SystemExit(f"Missing source checkpoint: {source_latest}")
run_dir = output_dir / "action_anticipation_frozen" / run_tag
run_dir.mkdir(parents=True, exist_ok=True)
latest = run_dir / "latest.pt"
if latest.exists() or latest.is_symlink():
    latest.unlink()
try:
    latest.symlink_to(source_latest)
except OSError:
    shutil.copy2(source_latest, latest)

cfg["nodes"] = 1
cfg["tasks_per_node"] = 1
cfg["cpus_per_task"] = 12
cfg["tag"] = run_tag
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
data["video_downsample_factor"] = float(os.environ.get("VIDEO_DOWNSAMPLE_FACTOR", "1") or "1")
data["num_workers"] = int(os.environ["EVAL_NUM_WORKERS"])
data["val_anticipation_point"] = [0.0, 0.0]

cfg["experiment"]["optimization"]["batch_size"] = int(os.environ["EVAL_BATCH_SIZE"])
cfg["experiment"]["optimization"]["num_epochs"] = 1
cfg["model_kwargs"]["checkpoint"] = os.environ["CHECKPOINT"]
cfg["model_kwargs"]["module_name"] = "app.hdepic_lora_action_anticipation.modelcustom.vit_encoder_predictor_rollout"
cfg["model_kwargs"]["wrapper_kwargs"]["num_steps"] = 1
cfg["model_kwargs"]["wrapper_kwargs"]["num_output_frames"] = 2
cfg["model_kwargs"]["wrapper_kwargs"]["max_rollout_steps"] = int(os.environ["MAX_ROLLOUT_STEPS"])

lora_cfg = {
    "enabled": True,
    "rank": int(os.environ["LORA_RANK"]),
    "alpha": float(os.environ["LORA_ALPHA"]),
    "dropout": float(os.environ["LORA_DROPOUT"]),
    "train_heads": True,
    "pretrained_probe": os.environ["LORA_PRETRAINED_PROBE"],
    "align_reference_metrics": True,
    "val_metric_scope": os.environ.get("METRIC_SCOPE", "native"),
}

gaze_mode = os.environ.get("GAZE_MODE", "none").lower()
if gaze_mode != "none":
    lora_cfg["gaze"] = {
        "mode": gaze_mode,
        "gaze_root": os.environ["GAZE_ROOT"],
        "extract_root": os.environ["GAZE_EXTRACT_ROOT"],
        "sync_root": os.environ["GAZE_SYNC_ROOT"],
        "binary_radius_px": float(os.environ.get("GAZE_BINARY_RADIUS_PX", "64")),
        "traj_len": int(os.environ.get("GAZE_RNN_TRAJ_LEN", "1024")),
        "input_adapter": {
            "hidden_dim": int(os.environ.get("BINARY_INPUT_ADAPTER_HIDDEN_DIM", "8")),
            "scale": float(os.environ.get("BINARY_INPUT_ADAPTER_SCALE", "1.0")),
            "temporal_kernel": int(os.environ.get("BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL", "1")),
            "residual_clamp": float(os.environ.get("BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP", "1.0")),
        },
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
        },
    }

cfg["experiment"]["lora"] = lora_cfg

out = Path(os.environ["CONFIG_PATH"])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[future-latents] wrote {out}")
print(f"[future-latents] source_tag={source_tag} run_tag={run_tag}")
PY

python -m app.hdepic_lora_action_anticipation.future_latent_compare \
    --config "${CONFIG_PATH}" \
    --horizons "${HORIZONS}" \
    --out "${RESULT_CSV}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${EVAL_NUM_WORKERS}" \
    --max-samples "${MAX_SAMPLES}" \
    --metric-scope "${METRIC_SCOPE}" \
    --head-selection "${HEAD_SELECTION}" \
    $(if [[ "${DROP_INCOMPLETE_HISTORY}" == "1" ]]; then echo "--drop-incomplete-history"; fi)
BASH
)

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
