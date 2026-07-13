#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_pastwin_train
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=640GB
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_pastwin_train_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_pastwin_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Train clean original-video LoRA baselines where each target action is
# supervised from a full input clip ending LABEL_HORIZON seconds before it.
# The model still receives PREDICTION_HORIZON as its predictor horizon, matching
# the past-window diagnostic path while allowing LoRA/heads to learn this mapping.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
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
LORA_TAG_PREFIX="${LORA_TAG_PREFIX:-}"
GAZE_MODE="${GAZE_MODE:-none}"
GAZE_ROOT="${GAZE_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/SLAM-and-Gaze}"
GAZE_EXTRACT_ROOT="${GAZE_EXTRACT_ROOT:-${PROJECT_ROOT}/outputs/hdepic_gaze_token_gate/extracted}"
GAZE_SYNC_ROOT="${GAZE_SYNC_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/Videos}"
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
GAZE_BINARY_RADIUS_PX="${GAZE_BINARY_RADIUS_PX:-64}"
GAZE_BINARY_MAP_TYPE="${GAZE_BINARY_MAP_TYPE:-binary}"
BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP="${BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP:-0}"
BINARY_INPUT_ADAPTER_AUG_AWARE="${BINARY_INPUT_ADAPTER_AUG_AWARE:-0}"
BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG="${BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG:-1}"
BINARY_INPUT_ADAPTER_HIDDEN_DIM="${BINARY_INPUT_ADAPTER_HIDDEN_DIM:-8}"
BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL="${BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL:-1}"
BINARY_INPUT_ADAPTER_SCALE="${BINARY_INPUT_ADAPTER_SCALE:-1.0}"
BINARY_INPUT_ADAPTER_BINARY_CENTER="${BINARY_INPUT_ADAPTER_BINARY_CENTER:-0.0}"
BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP="${BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP:-1.0}"
BINARY_INPUT_ADAPTER_LR_MULT="${BINARY_INPUT_ADAPTER_LR_MULT:-0.05}"
BINARY_INPUT_ADAPTER_WEIGHT_DECAY="${BINARY_INPUT_ADAPTER_WEIGHT_DECAY:-0.0001}"
BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING="${BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING:-0}"
EVAL_GPUS="${EVAL_GPUS:-1}"
EVAL_DEBUGMODE="${EVAL_DEBUGMODE:-auto}"
EVAL_CPUS_PER_TASK="${EVAL_CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-12}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-12}"
EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-}"
EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-5}"
EVAL_MAX_TRAIN_ITERS="${EVAL_MAX_TRAIN_ITERS:-0}"
EVAL_MAX_LR="${EVAL_MAX_LR:-}"
EVAL_LR_SCALE="${EVAL_LR_SCALE:-1}"
EVAL_CLIP_BALANCED="${EVAL_CLIP_BALANCED:-1}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"
DEBUG_SUBSET_PATH="${DEBUG_SUBSET_PATH:-}"
GAZE_DATALOADER_TRACE_DIR="${GAZE_DATALOADER_TRACE_DIR:-}"
GAZE_DATALOADER_TRACE_EVENTS="${GAZE_DATALOADER_TRACE_EVENTS:-0}"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
PREDICTION_DUMP="${PREDICTION_DUMP:-0}"
LORA_VAL_METRIC_SCOPE="${LORA_VAL_METRIC_SCOPE:-native}"

PERF_MONITOR="${PERF_MONITOR:-0}"
PERF_INTERVAL="${PERF_INTERVAL:-30}"

mkdir -p "${PROJECT_ROOT}/logs" "${CONFIG_DIR}" "${OUTPUT_DIR}"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export ADAPTED_ANNOTATIONS_DIR ADAPTED_VIDEO_ROOT CONFIG_DIR CHECKPOINT OUTPUT_DIR LORA_PRETRAINED_PROBE
export PREDICTION_HORIZONS LABEL_HORIZONS PAST_WINDOW_CURRICULUM LORA_TAG_PREFIX EVAL_GPUS EVAL_DEBUGMODE EVAL_CPUS_PER_TASK EVAL_BATCH_SIZE EVAL_NUM_WORKERS EVAL_VAL_NUM_WORKERS EVAL_NUM_EPOCHS EVAL_MAX_TRAIN_ITERS EVAL_MAX_LR EVAL_LR_SCALE EVAL_CLIP_BALANCED RESUME_CHECKPOINT DEBUG_SUBSET_PATH
export LORA_RANK LORA_ALPHA LORA_DROPOUT PREDICTION_DUMP LORA_VAL_METRIC_SCOPE
export GAZE_MODE GAZE_ROOT GAZE_EXTRACT_ROOT GAZE_SYNC_ROOT GAZE_RNN_HIDDEN_DIM GAZE_RNN_NUM_LAYERS GAZE_RNN_NUM_TOKENS GAZE_RNN_DROPOUT GAZE_RNN_BIDIR GAZE_RNN_TRAJ_LEN
export GAZE_RNN_USE_VIDEO_TOKEN GAZE_RNN_VIDEO_FUSION GAZE_RNN_VIDEO_PROJ_DIM GAZE_RNN_LOCAL_TEMPORAL_RADIUS GAZE_RNN_LOCAL_SPATIAL_RADIUS GAZE_RNN_RESIDUAL_ALPHA_INIT GAZE_RNN_HISTORY_SEC GAZE_LR_MULT
export GAZE_BINARY_RADIUS_PX GAZE_BINARY_MAP_TYPE BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP BINARY_INPUT_ADAPTER_AUG_AWARE BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG
export BINARY_INPUT_ADAPTER_HIDDEN_DIM BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL BINARY_INPUT_ADAPTER_SCALE BINARY_INPUT_ADAPTER_BINARY_CENTER BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP
export BINARY_INPUT_ADAPTER_LR_MULT BINARY_INPUT_ADAPTER_WEIGHT_DECAY BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING
export GAZE_DATALOADER_TRACE_DIR GAZE_DATALOADER_TRACE_EVENTS
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
        tag_prefix = os.environ.get("LORA_TAG_PREFIX", "").strip()
        if not tag_prefix:
            gaze_mode = os.environ.get("GAZE_MODE", "none")
            if gaze_mode == "none":
                tag_prefix = "hdepic-lora-pastwindow-clean"
            else:
                safe_mode = gaze_mode.replace("_", "-")
                tag_prefix = f"hdepic-lora-pastwindow-{safe_mode}"
        run_tag = f"{tag_prefix}-p{pred_tag}s-l{label_tag}s-train{suffix}"

        with template_path.open("r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        cfg["nodes"] = 1
        cfg["tasks_per_node"] = int(os.environ["EVAL_GPUS"])
        cfg["cpus_per_task"] = int(os.environ["EVAL_CPUS_PER_TASK"])
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
        if os.environ.get("EVAL_VAL_NUM_WORKERS", "").strip():
            data["val_num_workers"] = int(os.environ["EVAL_VAL_NUM_WORKERS"])
        data["clip_balanced"] = os.environ.get("EVAL_CLIP_BALANCED", "1") == "1"

        cfg["experiment"]["optimization"]["batch_size"] = int(os.environ["EVAL_BATCH_SIZE"])
        cfg["experiment"]["optimization"]["num_epochs"] = int(os.environ["EVAL_NUM_EPOCHS"])
        multihead_kwargs = cfg["experiment"]["optimization"]["multihead_kwargs"]
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
            print(f"[pastwin-train] Applied EVAL_LR_SCALE={lr_scale:g}; scaled {scaled} lr/start/ref/final lr values")
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
            print(f"[pastwin-train] Applied EVAL_MAX_LR={max_lr:g} after scaling; capped {capped} lr/start/ref lr values")
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

        gaze_mode = os.environ.get("GAZE_MODE", "none")
        cfg["experiment"]["lora"] = {
            "enabled": True,
            "rank": int(os.environ["LORA_RANK"]),
            "alpha": float(os.environ["LORA_ALPHA"]),
            "dropout": float(os.environ["LORA_DROPOUT"]),
            "train_heads": True,
            "pretrained_probe": os.environ["LORA_PRETRAINED_PROBE"],
            "align_reference_metrics": True,
            "val_metric_scope": os.environ.get("LORA_VAL_METRIC_SCOPE", "native"),
            "past_window_baseline": past_window_baseline,
            "gaze": {
                "mode": gaze_mode,
                "gaze_root": os.environ["GAZE_ROOT"],
                "extract_root": os.environ["GAZE_EXTRACT_ROOT"],
                "sync_root": os.environ["GAZE_SYNC_ROOT"],
                "traj_len": int(os.environ["GAZE_RNN_TRAJ_LEN"]),
                "rnn": {
                    "hidden_dim": int(os.environ["GAZE_RNN_HIDDEN_DIM"]),
                    "num_layers": int(os.environ["GAZE_RNN_NUM_LAYERS"]),
                    "bidirectional": os.environ["GAZE_RNN_BIDIR"] == "1",
                    "dropout": float(os.environ["GAZE_RNN_DROPOUT"]),
                    "num_tokens": int(os.environ["GAZE_RNN_NUM_TOKENS"]),
                    "use_video_tokens": os.environ["GAZE_RNN_USE_VIDEO_TOKEN"] == "1",
                    "video_fusion": os.environ["GAZE_RNN_VIDEO_FUSION"],
                    "video_proj_dim": int(os.environ["GAZE_RNN_VIDEO_PROJ_DIM"]),
                    "local_temporal_radius": int(os.environ["GAZE_RNN_LOCAL_TEMPORAL_RADIUS"]),
                    "local_spatial_radius": int(os.environ["GAZE_RNN_LOCAL_SPATIAL_RADIUS"]),
                    "residual_alpha_init": float(os.environ["GAZE_RNN_RESIDUAL_ALPHA_INIT"]),
                    "history_sec": float(os.environ["GAZE_RNN_HISTORY_SEC"]),
                    "gaze_lr_mult": float(os.environ["GAZE_LR_MULT"]),
                },
                "binary_radius_px": float(os.environ["GAZE_BINARY_RADIUS_PX"]),
                "binary_map_type": os.environ["GAZE_BINARY_MAP_TYPE"],
                "force_zero_map": os.environ["BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP"] == "1",
                "aug_aware": os.environ["BINARY_INPUT_ADAPTER_AUG_AWARE"] == "1",
                "disable_train_aug": os.environ["BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG"] == "1",
                "input_adapter": {
                    "hidden_dim": int(os.environ["BINARY_INPUT_ADAPTER_HIDDEN_DIM"]),
                    "scale": float(os.environ["BINARY_INPUT_ADAPTER_SCALE"]),
                    "temporal_kernel": int(os.environ["BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL"]),
                    "binary_center": float(os.environ["BINARY_INPUT_ADAPTER_BINARY_CENTER"]),
                    "residual_clamp": float(os.environ["BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP"]),
                    "lr_mult": float(os.environ["BINARY_INPUT_ADAPTER_LR_MULT"]),
                    "weight_decay": float(os.environ["BINARY_INPUT_ADAPTER_WEIGHT_DECAY"]),
                    "activation_checkpointing": os.environ["BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING"] == "1",
                },
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
        if [[ -n "${LORA_TAG_PREFIX}" ]]; then
            run_tag="${LORA_TAG_PREFIX}-p${pred_tag}s-l${label_tag}s-train"
        elif [[ "${GAZE_MODE}" == "none" ]]; then
            run_tag="hdepic-lora-pastwindow-clean-p${pred_tag}s-l${label_tag}s-train"
        else
            safe_mode="${GAZE_MODE//_/-}"
            run_tag="hdepic-lora-pastwindow-${safe_mode}-p${pred_tag}s-l${label_tag}s-train"
        fi
        if [[ "${PAST_WINDOW_CURRICULUM}" == "1" ]]; then
            run_tag="${run_tag}-curriculum"
        fi
        config_path="${CONFIG_DIR}/${run_tag}.yaml"
        echo "[pastwin-train] training ${run_tag}"
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
        eval_cmd=(python -m evals.main --fname "${config_path}" --devices "${device_args[@]}")
        if [[ "${debugmode}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
            eval_cmd+=(--debugmode true)
        fi
        echo "[pastwin-train] launch command: ${eval_cmd[*]}"
        "${eval_cmd[@]}"
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
    perf_prefix="${PROJECT_ROOT}/logs/perf_pastwin_train_${SLURM_JOB_ID:-manual}"
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
