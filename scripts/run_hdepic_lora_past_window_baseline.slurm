#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_past_window
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_past_window_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_past_window_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Baseline for long-horizon inputs without long-horizon prediction.
# For each target action, sample an input clip ending 10s/60s before that action
# when a full clip is available, but pass 1s/3.5s as the model's prediction
# horizon. Binary-map overlay is intentionally excluded because that branch is
# an early/mostly-original-video preprocessing baseline.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

ADAPTED_ANNOTATIONS_DIR="${ADAPTED_ANNOTATIONS_DIR:-${PROJECT_ROOT}/data/hdepic_vjepa_annotations}"
ADAPTED_VIDEO_ROOT="${ADAPTED_VIDEO_ROOT:-${PROJECT_ROOT}/data/hdepic_vjepa_videos}"
GAZE_OVERLAY_VIDEO_ROOT="${GAZE_OVERLAY_VIDEO_ROOT:-${PROJECT_ROOT}/data/hdepic_vjepa_videos_gaze_sigma96}"
CONFIG_DIR="${CONFIG_DIR:-${PROJECT_ROOT}/configs/generated/past_window_baseline}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vitg-384.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"
LORA_PRETRAINED_PROBE="${LORA_PRETRAINED_PROBE:-${PROJECT_ROOT}/outputs/ek100_vitg384_probe/action_anticipation_frozen/ek100-vitg16-384/latest.pt}"
TOKEN_SOURCE_TAG="${TOKEN_SOURCE_TAG:-hdepic-lora-token-gaze-clipbal-h100-bs2-w12-m640}"

METHODS="${METHODS:-clean overlay token rnn mlp}"
PREDICTION_HORIZONS="${PREDICTION_HORIZONS:-1 3.5}"
LABEL_HORIZONS="${LABEL_HORIZONS:-10 60}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
PREDICTION_DUMP="${PREDICTION_DUMP:-0}"
LORA_VAL_METRIC_SCOPE="${LORA_VAL_METRIC_SCOPE:-native}"

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
GAZE_LR_MULT="${GAZE_LR_MULT:-5.0}"
GAZE_HIDDEN_DUMP="${GAZE_HIDDEN_DUMP:-0}"

mkdir -p "${PROJECT_ROOT}/logs" "${CONFIG_DIR}"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export ADAPTED_ANNOTATIONS_DIR ADAPTED_VIDEO_ROOT GAZE_OVERLAY_VIDEO_ROOT CONFIG_DIR CHECKPOINT OUTPUT_DIR LORA_PRETRAINED_PROBE TOKEN_SOURCE_TAG
export METHODS PREDICTION_HORIZONS LABEL_HORIZONS EVAL_BATCH_SIZE EVAL_NUM_WORKERS LORA_RANK LORA_ALPHA LORA_DROPOUT PREDICTION_DUMP LORA_VAL_METRIC_SCOPE
export GAZE_ROOT GAZE_EXTRACT_ROOT GAZE_SYNC_ROOT GAZE_GATE_GAMMA GAZE_GATE_SIGMA_PX GAZE_LEARNABLE_GATE
export GAZE_RNN_HIDDEN_DIM GAZE_RNN_NUM_LAYERS GAZE_RNN_NUM_TOKENS GAZE_RNN_DROPOUT GAZE_RNN_BIDIR GAZE_RNN_TRAJ_LEN GAZE_LR_MULT GAZE_HIDDEN_DUMP

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
output_dir = Path(os.environ["OUTPUT_DIR"])
methods = [x.strip() for x in os.environ["METHODS"].replace(",", " ").split() if x.strip()]
prediction_horizons = [x.strip() for x in os.environ["PREDICTION_HORIZONS"].replace(",", " ").split() if x.strip()]
label_horizons = [x.strip() for x in os.environ["LABEL_HORIZONS"].replace(",", " ").split() if x.strip()]

method_cfg = {
    "clean": {
        "source_tag": "hdepic-lora-vitg16-384-cleanp01-refmetrics",
        "gaze_mode": "none",
        "video_root": os.environ["ADAPTED_VIDEO_ROOT"],
    },
    "overlay": {
        "source_tag": "hdepic-lora-overlay-gaze-sigma96-clipbal-h100-bs2-w12-m640",
        "gaze_mode": "overlay_video",
        "video_root": os.environ["GAZE_OVERLAY_VIDEO_ROOT"],
    },
    "token": {
        "source_tag": os.environ["TOKEN_SOURCE_TAG"],
        "gaze_mode": "token_gate",
        "video_root": os.environ["ADAPTED_VIDEO_ROOT"],
    },
    "rnn": {
        "source_tag": "hdepic-lora-rnn-gaze-clipbal-h100-bs2-w12-m640",
        "gaze_mode": "rnn_fuse",
        "video_root": os.environ["ADAPTED_VIDEO_ROOT"],
    },
    "mlp": {
        "source_tag": "hdepic-lora-mlp-gaze-clipbal-h100-bs2-w12-m640",
        "gaze_mode": "mlp_fuse",
        "video_root": os.environ["ADAPTED_VIDEO_ROOT"],
    },
}

def horizon_tag(value: str) -> str:
    f = float(value)
    return str(int(f)) if f.is_integer() else str(f).replace(".", "p")

for method in methods:
    if method not in method_cfg:
        raise SystemExit(f"Unknown method={method}; expected one of {sorted(method_cfg)}")
    m = method_cfg[method]
    source_tag = m["source_tag"]
    source_latest = output_dir / "action_anticipation_frozen" / source_tag / "latest.pt"
    if not source_latest.exists():
        raise SystemExit(f"Missing source checkpoint for {method}: {source_latest}")

    for pred_h in prediction_horizons:
        for label_h in label_horizons:
            pred_tag = horizon_tag(pred_h)
            label_tag = horizon_tag(label_h)
            run_tag = f"{source_tag}__pastwindow_p{pred_tag}s_l{label_tag}s"
            run_dir = output_dir / "action_anticipation_frozen" / run_tag
            run_dir.mkdir(parents=True, exist_ok=True)
            latest_link = run_dir / "latest.pt"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            try:
                latest_link.symlink_to(source_latest)
            except OSError:
                import shutil

                shutil.copy2(source_latest, latest_link)

            with template_path.open("r", encoding="utf-8") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

            cfg["nodes"] = 1
            cfg["tasks_per_node"] = 1
            cfg["cpus_per_task"] = 12
            cfg["tag"] = run_tag
            cfg["eval_name"] = "app.hdepic_lora_action_anticipation"
            cfg["folder"] = str(output_dir)
            cfg["resume_checkpoint"] = True
            cfg["val_only"] = True

            data = cfg["experiment"]["data"]
            data["dataset"] = "EK100"
            data["file_format"] = 1
            data["base_path"] = m["video_root"]
            data["dataset_train"] = str(Path(os.environ["ADAPTED_ANNOTATIONS_DIR"]) / "HD_EPIC_train_vjepa.csv")
            data["dataset_val"] = str(Path(os.environ["ADAPTED_ANNOTATIONS_DIR"]) / "HD_EPIC_val_vjepa.csv")
            data["anticipation_time_sec"] = [float(label_h), float(label_h)]
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
                "val_metric_scope": os.environ.get("LORA_VAL_METRIC_SCOPE", "native"),
                "past_window_baseline": {
                    "enabled": True,
                    "prediction_horizon_sec": float(pred_h),
                    "label_horizon_sec": float(label_h),
                    "drop_incomplete_history": True,
                },
                "gaze": {
                    "mode": m["gaze_mode"],
                    "gaze_root": os.environ["GAZE_ROOT"],
                    "extract_root": os.environ["GAZE_EXTRACT_ROOT"],
                    "sync_root": os.environ["GAZE_SYNC_ROOT"],
                    "gamma": float(os.environ["GAZE_GATE_GAMMA"]),
                    "sigma_px": float(os.environ["GAZE_GATE_SIGMA_PX"]),
                    "learnable_gate": os.environ.get("GAZE_LEARNABLE_GATE", "1") == "1",
                    "use_motion": m["gaze_mode"] == "token_gate",
                    "traj_len": int(os.environ["GAZE_RNN_TRAJ_LEN"]),
                    "rnn": {
                        "hidden_dim": int(os.environ["GAZE_RNN_HIDDEN_DIM"]),
                        "num_layers": int(os.environ["GAZE_RNN_NUM_LAYERS"]),
                        "bidirectional": os.environ["GAZE_RNN_BIDIR"] == "1",
                        "dropout": float(os.environ["GAZE_RNN_DROPOUT"]),
                        "num_tokens": int(os.environ["GAZE_RNN_NUM_TOKENS"]),
                        "gaze_lr_mult": float(os.environ["GAZE_LR_MULT"]),
                    },
                    "hidden_dump": {
                        "enabled": os.environ["GAZE_HIDDEN_DUMP"] == "1",
                        "max_batches": 4,
                    },
                },
                "prediction_dump": {
                    "enabled": os.environ["PREDICTION_DUMP"] == "1",
                    "path": str(run_dir / f"val_predictions_pastwindow_p{pred_tag}s_l{label_tag}s.csv"),
                },
            }

            out_path = config_dir / f"hdepic_lora_pastwindow_{method}_p{pred_tag}s_l{label_tag}s.yaml"
            with out_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(f"[past-window] wrote {out_path}")
PY

cd "${VJEPA_ROOT}"
for method in ${METHODS//,/ }; do
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
            config_path="${CONFIG_DIR}/hdepic_lora_pastwindow_${method}_p${pred_tag}s_l${label_tag}s.yaml"
            echo "[past-window] method=${method} model_prediction=${pred_h}s input_before_action=${label_h}s"
            python -m evals.main \
                --fname "${config_path}" \
                --devices cuda:0 \
                --debugmode true
        done
    done
done
BASH
)

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
