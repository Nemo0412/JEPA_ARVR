#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_val
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_lora_val_%j.out
#SBATCH --error=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_lora_val_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# Val-only horizon probe using the current LoRA checkpoint.
# Direct predictor horizons are limited by the predictor positional range. With
# the current ViT-g/384 config this limit is about 3.75s; 10s requires a new
# iterative or long-horizon path rather than a single direct predictor call.

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

ADAPTED_ANNOTATIONS_DIR="${ADAPTED_ANNOTATIONS_DIR:-${PROJECT_ROOT}/data/hdepic_vjepa_annotations}"
ADAPTED_VIDEO_ROOT="${ADAPTED_VIDEO_ROOT:-${PROJECT_ROOT}/data/hdepic_vjepa_videos}"
CONFIG_DIR="${CONFIG_DIR:-${PROJECT_ROOT}/configs/generated}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vitg-384.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"
LORA_PRETRAINED_PROBE="${LORA_PRETRAINED_PROBE:-${PROJECT_ROOT}/outputs/ek100_vitg384_probe/action_anticipation_frozen/ek100-vitg16-384/latest.pt}"
LORA_TAG="${LORA_TAG:-hdepic-lora-vitg16-384-cleanp01-refmetrics}"

HORIZONS="${HORIZONS:-3.5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
ALLOW_UNSUPPORTED_HORIZON="${ALLOW_UNSUPPORTED_HORIZON:-0}"
GAZE_MODE="${GAZE_MODE:-none}"
GAZE_ROOT="${GAZE_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/SLAM-and-Gaze}"
GAZE_EXTRACT_ROOT="${GAZE_EXTRACT_ROOT:-${PROJECT_ROOT}/outputs/hdepic_gaze_token_gate/extracted}"
GAZE_SYNC_ROOT="${GAZE_SYNC_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/HD-EPIC/Videos}"
GAZE_GATE_GAMMA="${GAZE_GATE_GAMMA:-0.7}"
GAZE_GATE_SIGMA_PX="${GAZE_GATE_SIGMA_PX:-40}"
PREDICTION_DUMP="${PREDICTION_DUMP:-0}"
PREDICTION_DUMP_PATH="${PREDICTION_DUMP_PATH:-}"

mkdir -p "${PROJECT_ROOT}/logs" "${CONFIG_DIR}"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export ADAPTED_ANNOTATIONS_DIR ADAPTED_VIDEO_ROOT CONFIG_DIR CHECKPOINT OUTPUT_DIR LORA_PRETRAINED_PROBE
export LORA_TAG HORIZONS EVAL_BATCH_SIZE EVAL_NUM_WORKERS LORA_RANK LORA_ALPHA LORA_DROPOUT ALLOW_UNSUPPORTED_HORIZON
export GAZE_MODE GAZE_ROOT GAZE_EXTRACT_ROOT GAZE_SYNC_ROOT GAZE_GATE_GAMMA GAZE_GATE_SIGMA_PX
export PREDICTION_DUMP PREDICTION_DUMP_PATH

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

project_root = Path(os.environ["PROJECT_ROOT"])
template_path = project_root / "vjepa2" / "configs" / "eval" / "vitg-384" / "ek100.yaml"
config_dir = Path(os.environ["CONFIG_DIR"])
horizons = [h.strip() for h in os.environ["HORIZONS"].replace(",", " ").split() if h.strip()]

for horizon in horizons:
    h = float(horizon)
    h_tag = str(int(h)) if h.is_integer() else str(h).replace(".", "p")
    with template_path.open("r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    data_template = cfg["experiment"]["data"]
    wrapper = cfg["model_kwargs"].get("wrapper_kwargs", {})
    encoder = cfg["model_kwargs"]["pretrain_kwargs"]["encoder"]
    predictor = cfg["model_kwargs"]["pretrain_kwargs"]["predictor"]
    frames_per_clip = int(data_template["frames_per_clip"])
    fps = float(data_template["frames_per_second"])
    resolution = int(data_template["resolution"])
    patch_size = int(encoder.get("patch_size", 16))
    tubelet_size = int(encoder.get("tubelet_size", 2))
    predictor_num_frames = int(predictor.get("num_frames", frames_per_clip))
    num_output_frames = max(int(wrapper.get("num_output_frames", tubelet_size)), tubelet_size)

    grid = resolution // patch_size
    spatial_tokens = grid * grid
    context_tokens = (frames_per_clip // tubelet_size) * spatial_tokens
    predictor_tokens = (predictor_num_frames // tubelet_size) * spatial_tokens
    pred_tokens = (num_output_frames // tubelet_size) * spatial_tokens
    max_steps = (predictor_tokens - context_tokens - pred_tokens) // spatial_tokens
    max_direct_horizon = max_steps * tubelet_size / fps

    if h > max_direct_horizon and os.environ.get("ALLOW_UNSUPPORTED_HORIZON") != "1":
        raise SystemExit(
            "[val] Unsupported direct horizon "
            f"{h}s for this predictor config. Max direct horizon is about "
            f"{max_direct_horizon:.2f}s "
            f"(frames_per_clip={frames_per_clip}, fps={fps:g}, "
            f"tubelet={tubelet_size}, predictor_num_frames={predictor_num_frames}, "
            f"num_output_frames={num_output_frames}). "
            "Use HORIZONS<=max for direct validation, or implement iterative/long-horizon prediction."
        )

    cfg["nodes"] = 1
    cfg["tasks_per_node"] = 1
    cfg["cpus_per_task"] = 12
    cfg["tag"] = os.environ["LORA_TAG"]
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
    data["anticipation_time_sec"] = [h, h]
    data["num_workers"] = int(os.environ["EVAL_NUM_WORKERS"])

    cfg["experiment"]["optimization"]["batch_size"] = int(os.environ["EVAL_BATCH_SIZE"])
    cfg["experiment"]["optimization"]["num_epochs"] = 1
    cfg["model_kwargs"]["checkpoint"] = os.environ["CHECKPOINT"]
    cfg["experiment"]["lora"] = {
        "enabled": True,
        "rank": int(os.environ["LORA_RANK"]),
        "alpha": float(os.environ["LORA_ALPHA"]),
        "dropout": float(os.environ["LORA_DROPOUT"]),
        "train_heads": True,
        "pretrained_probe": os.environ["LORA_PRETRAINED_PROBE"],
        "align_reference_metrics": True,
        "gaze": {
            "mode": os.environ["GAZE_MODE"],
            "gaze_root": os.environ["GAZE_ROOT"],
            "extract_root": os.environ["GAZE_EXTRACT_ROOT"],
            "sync_root": os.environ["GAZE_SYNC_ROOT"],
            "gamma": float(os.environ["GAZE_GATE_GAMMA"]),
            "sigma_px": float(os.environ["GAZE_GATE_SIGMA_PX"]),
            "use_motion": True,
        },
        "prediction_dump": {
            "enabled": os.environ["PREDICTION_DUMP"] == "1",
            **({"path": os.environ["PREDICTION_DUMP_PATH"]} if os.environ.get("PREDICTION_DUMP_PATH") else {}),
        },
    }

    out_path = config_dir / f"hdepic_lora_valonly_{h_tag}s.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[val] Wrote {out_path}")
PY

cd "${VJEPA_ROOT}"
for horizon in ${HORIZONS//,/ }; do
    h_tag="${horizon}"
    if [[ "${h_tag}" == *.0 ]]; then
        h_tag="${h_tag%.0}"
    else
        h_tag="${h_tag//./p}"
    fi
    config_path="${CONFIG_DIR}/hdepic_lora_valonly_${h_tag}s.yaml"
    echo "[val] Running LoRA checkpoint at horizon ${horizon}s"
    python -m evals.main \
        --fname "${config_path}" \
        --devices cuda:0 \
        --debugmode true
done
BASH
)

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
