#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_pastwin_train
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=640GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_lora_pastwin_train_%j.out
#SBATCH --error=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_lora_pastwin_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# Train clean original-video LoRA baselines where each target action is
# supervised from a full input clip ending LABEL_HORIZON seconds before it.
# The model still receives PREDICTION_HORIZON as its predictor horizon, matching
# the past-window diagnostic path while allowing LoRA/heads to learn this mapping.

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

ADAPTED_ANNOTATIONS_DIR="${ADAPTED_ANNOTATIONS_DIR:-${PROJECT_ROOT}/data/hdepic_vjepa_annotations}"
ADAPTED_VIDEO_ROOT="${ADAPTED_VIDEO_ROOT:-${PROJECT_ROOT}/data/hdepic_vjepa_videos}"
CONFIG_DIR="${CONFIG_DIR:-${PROJECT_ROOT}/configs/generated/past_window_train}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vitg-384.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"
LORA_PRETRAINED_PROBE="${LORA_PRETRAINED_PROBE:-${PROJECT_ROOT}/outputs/ek100_vitg384_probe/action_anticipation_frozen/ek100-vitg16-384/latest.pt}"

PREDICTION_HORIZONS="${PREDICTION_HORIZONS:-1}"
LABEL_HORIZONS="${LABEL_HORIZONS:-10 60}"
PAST_WINDOW_CURRICULUM="${PAST_WINDOW_CURRICULUM:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-12}"
EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-5}"
EVAL_CLIP_BALANCED="${EVAL_CLIP_BALANCED:-1}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
PREDICTION_DUMP="${PREDICTION_DUMP:-0}"

PERF_MONITOR="${PERF_MONITOR:-0}"
PERF_INTERVAL="${PERF_INTERVAL:-30}"

mkdir -p "${PROJECT_ROOT}/logs" "${CONFIG_DIR}" "${OUTPUT_DIR}"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export ADAPTED_ANNOTATIONS_DIR ADAPTED_VIDEO_ROOT CONFIG_DIR CHECKPOINT OUTPUT_DIR LORA_PRETRAINED_PROBE
export PREDICTION_HORIZONS LABEL_HORIZONS PAST_WINDOW_CURRICULUM EVAL_BATCH_SIZE EVAL_NUM_WORKERS EVAL_NUM_EPOCHS EVAL_CLIP_BALANCED RESUME_CHECKPOINT
export LORA_RANK LORA_ALPHA LORA_DROPOUT PREDICTION_DUMP
export PERF_MONITOR PERF_INTERVAL

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

if [[ ! -f "${PROJECT_ROOT}/app/hdepic_lora_action_anticipation/eval.py" ]]; then
    echo "[pastwin-train] Missing project eval module; sync app/ before running." >&2
    exit 1
fi

python - <<'PY'
import importlib
import sys

print("[pastwin-train] sys.path head:", sys.path[:5])
importlib.import_module("app.hdepic_lora_action_anticipation.eval")
print("[pastwin-train] app.hdepic_lora_action_anticipation import OK")
PY

python - <<'PY'
import os
from pathlib import Path
import yaml

project_root = Path(os.environ["PROJECT_ROOT"])
template_path = project_root / "vjepa2" / "configs" / "eval" / "vitg-384" / "ek100.yaml"
config_dir = Path(os.environ["CONFIG_DIR"])
prediction_horizons = [x.strip() for x in os.environ["PREDICTION_HORIZONS"].replace(",", " ").split() if x.strip()]
label_horizons = [x.strip() for x in os.environ["LABEL_HORIZONS"].replace(",", " ").split() if x.strip()]


def horizon_tag(value: str) -> str:
    f = float(value)
    return str(int(f)) if f.is_integer() else str(f).replace(".", "p")


for pred_h in prediction_horizons:
    for label_h in label_horizons:
        pred_tag = horizon_tag(pred_h)
        label_tag = horizon_tag(label_h)
        curriculum_enabled = os.environ.get("PAST_WINDOW_CURRICULUM", "0") == "1"
        suffix = "-curriculum" if curriculum_enabled else ""
        run_tag = f"hdepic-lora-pastwindow-clean-p{pred_tag}s-l{label_tag}s-train{suffix}"

        with template_path.open("r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        cfg["nodes"] = 1
        cfg["tasks_per_node"] = 1
        cfg["cpus_per_task"] = 12
        cfg["tag"] = run_tag
        cfg["eval_name"] = "app.hdepic_lora_action_anticipation"
        cfg["folder"] = os.environ["OUTPUT_DIR"]
        cfg["resume_checkpoint"] = os.environ["RESUME_CHECKPOINT"] == "1"
        cfg["val_only"] = False

        data = cfg["experiment"]["data"]
        data["dataset"] = "EK100"
        data["file_format"] = 1
        data["base_path"] = os.environ["ADAPTED_VIDEO_ROOT"]
        data["dataset_train"] = str(Path(os.environ["ADAPTED_ANNOTATIONS_DIR"]) / "HD_EPIC_train_vjepa.csv")
        data["dataset_val"] = str(Path(os.environ["ADAPTED_ANNOTATIONS_DIR"]) / "HD_EPIC_val_vjepa.csv")
        data["train_anticipation_time_sec"] = [float(label_h), float(label_h)]
        data["anticipation_time_sec"] = [float(label_h), float(label_h)]
        data["num_workers"] = int(os.environ["EVAL_NUM_WORKERS"])
        data["clip_balanced"] = os.environ.get("EVAL_CLIP_BALANCED", "1") == "1"

        cfg["experiment"]["optimization"]["batch_size"] = int(os.environ["EVAL_BATCH_SIZE"])
        cfg["experiment"]["optimization"]["num_epochs"] = int(os.environ["EVAL_NUM_EPOCHS"])
        cfg["model_kwargs"]["checkpoint"] = os.environ["CHECKPOINT"]
        past_window_baseline = {
            "enabled": True,
            "prediction_horizon_sec": float(pred_h),
            "label_horizon_sec": float(label_h),
            "drop_incomplete_history": True,
            "apply_to_train": True,
        }
        if curriculum_enabled:
            past_window_baseline["curriculum"] = {
                "enabled": True,
                "stages": [
                    {"until_epoch": 1, "label_horizon_sec": [0.25, 2.0]},
                    {"until_epoch": 2, "label_horizon_sec": [2.0, 5.0]},
                    {"until_epoch": 3, "label_horizon_sec": [5.0, float(label_h)]},
                    {"label_horizon_sec": [float(label_h), float(label_h)]},
                ],
            }

        cfg["experiment"]["lora"] = {
            "enabled": True,
            "rank": int(os.environ["LORA_RANK"]),
            "alpha": float(os.environ["LORA_ALPHA"]),
            "dropout": float(os.environ["LORA_DROPOUT"]),
            "train_heads": True,
            "pretrained_probe": os.environ["LORA_PRETRAINED_PROBE"],
            "align_reference_metrics": True,
            "past_window_baseline": past_window_baseline,
            "gaze": {
                "mode": "none",
            },
            "prediction_dump": {
                "enabled": os.environ["PREDICTION_DUMP"] == "1",
            },
        }

        out_path = config_dir / f"{run_tag}.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"[pastwin-train] wrote {out_path}")
PY

cd "${VJEPA_ROOT}"
for pred_h in ${PREDICTION_HORIZONS//,/ }; do
    pred_tag="${pred_h}"
    if [[ "${pred_tag}" == *.0 ]]; then
        pred_tag="${pred_tag%.0}"
    else
        pred_tag="${pred_tag//./p}"
    fi
    for label_h in ${LABEL_HORIZONS//,/ }; do
        label_tag="${label_h}"
        if [[ "${label_tag}" == *.0 ]]; then
            label_tag="${label_tag%.0}"
        else
            label_tag="${label_tag//./p}"
        fi
        run_tag="hdepic-lora-pastwindow-clean-p${pred_tag}s-l${label_tag}s-train"
        if [[ "${PAST_WINDOW_CURRICULUM}" == "1" ]]; then
            run_tag="${run_tag}-curriculum"
        fi
        config_path="${CONFIG_DIR}/${run_tag}.yaml"
        echo "[pastwin-train] training ${run_tag}"
        python -m evals.main \
            --fname "${config_path}" \
            --devices cuda:0 \
            --debugmode true
    done
done
BASH
)

monitor_pids=()
stop_perf_monitors() {
    for pid in "${monitor_pids[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}

if [[ "${PERF_MONITOR}" == "1" ]]; then
    perf_prefix="${PROJECT_ROOT}/logs/perf_pastwin_train_${SLURM_JOB_ID:-manual}"
    echo "[perf] Sampling nvidia-smi every ${PERF_INTERVAL}s into ${perf_prefix}_nvidia_smi.csv"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
            --format=csv -l "${PERF_INTERVAL}" > "${perf_prefix}_nvidia_smi.csv" 2> "${perf_prefix}_nvidia_smi.err" &
        monitor_pids+=("$!")
    fi
    trap stop_perf_monitors EXIT
fi

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
