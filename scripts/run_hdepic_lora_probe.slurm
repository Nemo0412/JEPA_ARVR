#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_probe
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_lora_probe_%j.out
#SBATCH --error=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_lora_probe_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# Sidecar LoRA finetune path for the action anticipation probe/decoder.
# Does not edit vjepa2/. Uses eval_name=app.hdepic_lora_action_anticipation.
# Default path uses the official EK100 ViT-g/384 probe as the decoder starting
# point because its pooler dimensions match the official vitg-384 encoder.

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"
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
LORA_PRETRAINED_PROBE="${LORA_PRETRAINED_PROBE:-${PROJECT_ROOT}/outputs/ek100_vitg384_probe/action_anticipation_frozen/ek100-vitg16-384/latest.pt}"
LORA_TAG="${LORA_TAG:-hdepic-lora-vitg16-384-cleanp01-refmetrics}"
GAZE_MODE="${GAZE_MODE:-none}"
SIGMA="${SIGMA:-96}"
GAZE_VIDEO_ROOT="${GAZE_VIDEO_ROOT:-${PROJECT_ROOT}/data/hdepic_vjepa_videos_gaze_sigma${SIGMA}}"
GAZE_ROOT="${GAZE_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/SLAM-and-Gaze}"
GAZE_EXTRACT_ROOT="${GAZE_EXTRACT_ROOT:-${PROJECT_ROOT}/outputs/hdepic_gaze_token_gate/extracted}"
GAZE_SYNC_ROOT="${GAZE_SYNC_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/HD-EPIC/Videos}"
GAZE_GATE_GAMMA="${GAZE_GATE_GAMMA:-0.7}"
GAZE_GATE_SIGMA_PX="${GAZE_GATE_SIGMA_PX:-40}"
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
GAZE_LR_MULT="${GAZE_LR_MULT:-5.0}"
GAZE_HIDDEN_DUMP="${GAZE_HIDDEN_DUMP:-0}"
PREDICTION_DUMP="${PREDICTION_DUMP:-0}"
PREDICTION_DUMP_PATH="${PREDICTION_DUMP_PATH:-}"
PERF_MONITOR="${PERF_MONITOR:-0}"
PERF_INTERVAL="${PERF_INTERVAL:-30}"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
EVAL_MAX_TRAIN_ITERS="${EVAL_MAX_TRAIN_ITERS:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"
VAL_ONLY="${VAL_ONLY:-0}"
BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP="${BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP:-0}"
BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT="${BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT:-}"

mkdir -p "${PROJECT_ROOT}/logs" "${CONFIG_DIR}" "${OUTPUT_DIR}"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export ADAPTED_ANNOTATIONS_DIR ADAPTED_VIDEO_ROOT CONFIG_DIR CONFIG_PATH CHECKPOINT OUTPUT_DIR LORA_PRETRAINED_PROBE
export LORA_TAG LORA_RANK LORA_ALPHA LORA_DROPOUT EVAL_BATCH_SIZE EVAL_NUM_EPOCHS EVAL_NUM_WORKERS EVAL_MAX_TRAIN_ITERS RESUME_CHECKPOINT VAL_ONLY
export GAZE_MODE SIGMA GAZE_VIDEO_ROOT GAZE_ROOT GAZE_EXTRACT_ROOT GAZE_SYNC_ROOT GAZE_GATE_GAMMA GAZE_GATE_SIGMA_PX
export GAZE_RNN_HIDDEN_DIM GAZE_RNN_NUM_LAYERS GAZE_RNN_NUM_TOKENS GAZE_RNN_DROPOUT GAZE_RNN_BIDIR GAZE_RNN_TRAJ_LEN
export GAZE_RNN_USE_VIDEO_TOKEN GAZE_RNN_VIDEO_FUSION GAZE_RNN_VIDEO_PROJ_DIM GAZE_RNN_LOCAL_TEMPORAL_RADIUS GAZE_RNN_LOCAL_SPATIAL_RADIUS GAZE_RNN_RESIDUAL_ALPHA_INIT GAZE_LR_MULT GAZE_HIDDEN_DUMP
export BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT
export PREDICTION_DUMP PREDICTION_DUMP_PATH
export PERF_MONITOR PERF_INTERVAL

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

case "${GAZE_MODE}" in
  none|token_gate|rnn_fuse|mlp_fuse|binary_input_adapter)
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
    echo "[lora] Unknown GAZE_MODE=${GAZE_MODE}; expected none, overlay_video, token_gate, rnn_fuse, mlp_fuse, or binary_input_adapter" >&2
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
import sys

print("[lora] sys.path head:", sys.path[:5])
importlib.import_module("app.hdepic_lora_action_anticipation.eval")
print("[lora] app.hdepic_lora_action_anticipation import OK")
PY

python - <<'PY'
import os
from pathlib import Path
import yaml

project_root = Path(os.environ["PROJECT_ROOT"])
template_path = project_root / "vjepa2" / "configs" / "eval" / "vitg-384" / "ek100.yaml"
config_path = Path(os.environ["CONFIG_PATH"])

with template_path.open("r", encoding="utf-8") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg["nodes"] = 1
cfg["tasks_per_node"] = 1
cfg["cpus_per_task"] = 12
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
data["num_workers"] = int(os.environ["EVAL_NUM_WORKERS"])
data["clip_balanced"] = os.environ.get("EVAL_CLIP_BALANCED", "1") == "1"

cfg["experiment"]["optimization"]["batch_size"] = int(os.environ["EVAL_BATCH_SIZE"])
cfg["experiment"]["optimization"]["num_epochs"] = int(os.environ["EVAL_NUM_EPOCHS"])
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
            "gaze_lr_mult": float(os.environ.get("GAZE_LR_MULT", "5.0")),
        },
        "hidden_dump": {
            "enabled": os.environ.get("GAZE_HIDDEN_DUMP", "0") == "1",
            "max_batches": 4,
        },
        "binary_radius_px": float(os.environ.get("GAZE_BINARY_RADIUS_PX", "64")),
        "fallback_full_frame": os.environ.get("GAZE_BINARY_FALLBACK_FULL_FRAME", "0") == "1",
        "force_zero_map": os.environ.get("BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP", "0") == "1",
        "disable_train_aug": os.environ.get(
            "BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG",
            "1" if os.environ["GAZE_MODE"] == "binary_input_adapter" else "0",
        ) == "1",
        "input_adapter": {
            "hidden_dim": int(os.environ.get("BINARY_INPUT_ADAPTER_HIDDEN_DIM", "8")),
            "scale": float(os.environ.get("BINARY_INPUT_ADAPTER_SCALE", "1.0")),
            "temporal_kernel": int(os.environ.get("BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL", "1")),
            "binary_center": float(os.environ.get("BINARY_INPUT_ADAPTER_BINARY_CENTER", "0.0")),
            "lr_mult": float(os.environ.get("BINARY_INPUT_ADAPTER_LR_MULT", "0.05")),
            "weight_decay": float(os.environ.get("BINARY_INPUT_ADAPTER_WEIGHT_DECAY", "0.0001")),
            "activation_checkpointing": os.environ.get("BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING", "0") == "1",
            **({"load_checkpoint_path": os.environ["BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT"]} if os.environ.get("BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT") else {}),
        },
    },
    "prediction_dump": {
        "enabled": os.environ["PREDICTION_DUMP"] == "1",
        **({"path": os.environ["PREDICTION_DUMP_PATH"]} if os.environ.get("PREDICTION_DUMP_PATH") else {}),
    },
}

config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"[lora] Wrote config: {config_path}")
PY

cd "${VJEPA_ROOT}"
python -m evals.main \
    --fname "${CONFIG_PATH}" \
    --devices cuda:0 \
    --debugmode true
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

sample_cgroup_memory() {
    local out_path="$1"
    local interval="$2"
    local cg_rel=""
    local cg_base=""
    local mem_current_file=""
    local mem_peak_file=""
    local mem_max_file=""

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

    {
        echo "timestamp,cgroup_path,current_mib,peak_mib,limit_mib,slurm_mem_per_node_mb"
        if [[ -z "${mem_current_file}" ]]; then
            echo "$(date '+%Y-%m-%dT%H:%M:%S%z'),not_found,,,,${SLURM_MEM_PER_NODE:-}"
            return 0
        fi
        while true; do
            local current_bytes=""
            local peak_bytes=""
            local limit_bytes=""
            local current_mib=""
            local peak_mib=""
            local limit_mib=""
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
    } > "${out_path}" 2> "${out_path%.csv}.err"
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

singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
