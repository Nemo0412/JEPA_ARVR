#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_aa_prep
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/hdepic_aa_prep_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/hdepic_aa_prep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# One-command HD-EPIC action anticipation workflow.
#
# Submit from the HPC login node:
#   sbatch /path/to/VJEPA2-EXP/scripts/run_hdepic_action_anticipation.slurm
#
# Optional overrides:
#   sbatch --export=ALL,DOWNLOAD_PARTICIPANTS=1,2,3,VAL_PARTICIPANTS=P03 \
#     /path/to/VJEPA2-EXP/scripts/run_hdepic_action_anticipation.slurm

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
RAW_DATA_PARENT="${RAW_DATA_PARENT:-${DATA_ROOT}/raw}"
HD_EPIC_ROOT="${HD_EPIC_ROOT:-${RAW_DATA_PARENT}/HD-EPIC}"
ANNOTATIONS_REPO="${ANNOTATIONS_REPO:-${DATA_ROOT}/hd-epic-annotations}"
DOWNLOADER_REPO="${DOWNLOADER_REPO:-${DATA_ROOT}/hd-epic-downloader}"
ADAPTED_ANNOTATIONS_DIR="${ADAPTED_ANNOTATIONS_DIR:-${DATA_ROOT}/hdepic_vjepa_annotations}"
ADAPTED_VIDEO_ROOT="${ADAPTED_VIDEO_ROOT:-${DATA_ROOT}/hdepic_vjepa_videos}"
CONFIG_DIR="${CONFIG_DIR:-${PROJECT_ROOT}/configs/generated}"
CONFIG_PATH="${CONFIG_PATH:-${CONFIG_DIR}/hdepic_action_anticipation_vitl384.yaml}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_action_anticipation}"

DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
DOWNLOAD_PARTICIPANTS="${DOWNLOAD_PARTICIPANTS:-1,2}"
INCLUDE_PARTICIPANTS="${INCLUDE_PARTICIPANTS:-P01}"
VAL_PARTICIPANTS="${VAL_PARTICIPANTS-P02}"
VAL_RATIO="${VAL_RATIO:-0.2}"
HDEPIC_SPLIT_PRESET="${HDEPIC_SPLIT_PRESET:-p01_fixed}"
CONVERT_FPS="${CONVERT_FPS:-30}"
LINK_METHOD="${LINK_METHOD:-symlink}"

EVAL_PARTITION="${EVAL_PARTITION:-}"
EVAL_QOS="${EVAL_QOS:-}"
EVAL_GPUS="${EVAL_GPUS:-1}"
EVAL_CPUS_PER_TASK="${EVAL_CPUS_PER_TASK:-12}"
EVAL_MEM="${EVAL_MEM:-128GB}"
EVAL_TIME="${EVAL_TIME:-48:00:00}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-20}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
SUBMIT_EVAL="${SUBMIT_EVAL:-1}"
MODE="${MODE:-prep}"

mkdir -p "${PROJECT_ROOT}" "${DATA_ROOT}" "${RAW_DATA_PARENT}" "${CONFIG_DIR}" "${EVAL_OUTPUT_DIR}" "${PROJECT_ROOT}/logs"

log_header() {
    echo "========================================"
    echo "Mode       : ${MODE}"
    echo "Job ID     : ${SLURM_JOB_ID:-not-in-slurm}"
    echo "Node       : $(hostname)"
    echo "Start      : $(date)"
    echo "Project    : ${PROJECT_ROOT}"
    echo "========================================"
}

run_in_container() {
    local nv_flag="$1"
    local payload="$2"

    if [[ "${nv_flag}" == "with_nv" ]]; then
        singularity exec --nv \
            --overlay "${OVERLAY}:ro" \
            "${SIF}" \
            /bin/bash -lc "${payload}"
    else
        singularity exec \
            --overlay "${OVERLAY}:ro" \
            "${SIF}" \
            /bin/bash -lc "${payload}"
    fi
}

prep_payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}

cd "${PROJECT_ROOT}"
mkdir -p "${DATA_ROOT}" "${RAW_DATA_PARENT}" "${ADAPTED_ANNOTATIONS_DIR}" "${ADAPTED_VIDEO_ROOT}" "${CONFIG_DIR}" "${EVAL_OUTPUT_DIR}"

echo "[prep] Python: $(which python)"
echo "[prep] Project root: ${PROJECT_ROOT}"

if [[ ! -d "${DOWNLOADER_REPO}/.git" ]]; then
    echo "[prep] Cloning HD-EPIC downloader"
    git clone --depth 1 https://github.com/hd-epic/hd-epic-downloader.git "${DOWNLOADER_REPO}"
else
    echo "[prep] Updating HD-EPIC downloader"
    git -C "${DOWNLOADER_REPO}" pull --ff-only
fi

if [[ ! -d "${ANNOTATIONS_REPO}/.git" ]]; then
    echo "[prep] Cloning HD-EPIC annotations"
    git clone --depth 1 https://github.com/hd-epic/hd-epic-annotations.git "${ANNOTATIONS_REPO}"
else
    echo "[prep] Updating HD-EPIC annotations"
    git -C "${ANNOTATIONS_REPO}" pull --ff-only
fi

if command -v git-lfs >/dev/null 2>&1; then
    git -C "${ANNOTATIONS_REPO}" lfs pull || true
fi

if [[ "${DOWNLOAD_DATA}" == "1" ]]; then
    downloader_args=("${RAW_DATA_PARENT}" "--videos")
    if [[ -n "${DOWNLOAD_PARTICIPANTS}" ]]; then
        downloader_args+=("--participants" "${DOWNLOAD_PARTICIPANTS}")
    fi
    echo "[prep] Downloading HD-EPIC videos. This is resumable."
    (
        cd "${DOWNLOADER_REPO}"
        python hd-epic-downloader.py "${downloader_args[@]}"
    )
else
    echo "[prep] DOWNLOAD_DATA=${DOWNLOAD_DATA}; skipping video download"
fi

ANNOTATIONS_PKL="${ANNOTATIONS_REPO}/narrations-and-action-segments/HD_EPIC_Narrations.pkl"
if [[ ! -f "${ANNOTATIONS_PKL}" ]]; then
    ANNOTATIONS_PKL="$(find "${ANNOTATIONS_REPO}" -name 'HD_EPIC_Narrations.pkl' -print -quit)"
fi
if [[ -z "${ANNOTATIONS_PKL}" || ! -f "${ANNOTATIONS_PKL}" ]]; then
    echo "[prep] Could not find HD_EPIC_Narrations.pkl under ${ANNOTATIONS_REPO}" >&2
    exit 1
fi

echo "[prep] Converting annotations from ${ANNOTATIONS_PKL}"
convert_args=(
    --annotations-pkl "${ANNOTATIONS_PKL}"
    --video-root "${HD_EPIC_ROOT}"
    --output-dir "${ADAPTED_ANNOTATIONS_DIR}"
    --link-root "${ADAPTED_VIDEO_ROOT}"
    --link-method "${LINK_METHOD}"
    --skip-missing-videos
    --no-video-probe
    --fps "${CONVERT_FPS}"
    --split-preset "${HDEPIC_SPLIT_PRESET}"
)
if [[ "${HDEPIC_SPLIT_PRESET}" == "legacy" && -n "${VAL_PARTICIPANTS}" ]]; then
    convert_args+=(--val-participants ${VAL_PARTICIPANTS})
elif [[ "${HDEPIC_SPLIT_PRESET}" == "legacy" ]]; then
    convert_args+=(--val-ratio "${VAL_RATIO}")
fi
if [[ -n "${INCLUDE_PARTICIPANTS}" ]]; then
    convert_args+=(--include-participants ${INCLUDE_PARTICIPANTS})
fi
python "${PROJECT_ROOT}/scripts/convert_hdepic_to_vjepa_csv.py" "${convert_args[@]}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[prep] Missing checkpoint: ${CHECKPOINT}" >&2
    echo "[prep] Put the V-JEPA2.1 ViT-L 384 checkpoint there or submit with CHECKPOINT=/path/to/checkpoint.pt" >&2
    exit 1
fi

python - <<'PY'
import os
from pathlib import Path
import yaml

project_root = Path(os.environ["PROJECT_ROOT"])
template_path = project_root / "vjepa2" / "configs" / "eval_2_1" / "vitl-384" / "ek100.yaml"
config_path = Path(os.environ["CONFIG_PATH"])

with template_path.open("r", encoding="utf-8") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg["nodes"] = 1
cfg["tasks_per_node"] = 1
cfg["cpus_per_task"] = int(os.environ["EVAL_CPUS_PER_TASK"])
cfg["tag"] = "hdepic-vitl16-384"
cfg["folder"] = os.environ["EVAL_OUTPUT_DIR"]
cfg["resume_checkpoint"] = True

data = cfg["experiment"]["data"]
data["dataset"] = "EK100"
data["file_format"] = 1
data["base_path"] = os.environ["ADAPTED_VIDEO_ROOT"]
data["dataset_train"] = str(Path(os.environ["ADAPTED_ANNOTATIONS_DIR"]) / "HD_EPIC_train_vjepa.csv")
data["dataset_val"] = str(Path(os.environ["ADAPTED_ANNOTATIONS_DIR"]) / "HD_EPIC_val_vjepa.csv")
data["dataset_test"] = str(Path(os.environ["ADAPTED_ANNOTATIONS_DIR"]) / "HD_EPIC_test_vjepa.csv")
data["num_workers"] = int(os.environ["EVAL_NUM_WORKERS"])

cfg["experiment"]["optimization"]["batch_size"] = int(os.environ["EVAL_BATCH_SIZE"])
cfg["experiment"]["optimization"]["num_epochs"] = int(os.environ["EVAL_NUM_EPOCHS"])
cfg["model_kwargs"]["checkpoint"] = os.environ["CHECKPOINT"]

config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"[prep] Wrote eval config: {config_path}")
PY

echo "[prep] Done"
BASH
)

eval_payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}

cd "${VJEPA_ROOT}"
echo "[eval] Python: $(which python)"
echo "[eval] Config: ${CONFIG_PATH}"
echo "[eval] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -m evals.main \
    --fname "${CONFIG_PATH}" \
    --devices cuda:0 \
    --debugmode true
BASH
)

log_header

if [[ ! -f "${SIF}" ]]; then
    echo "Missing Singularity image: ${SIF}" >&2
    exit 1
fi
if [[ ! -f "${OVERLAY}" ]]; then
    echo "Missing overlay: ${OVERLAY}" >&2
    exit 1
fi

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export DATA_ROOT RAW_DATA_PARENT HD_EPIC_ROOT ANNOTATIONS_REPO DOWNLOADER_REPO
export ADAPTED_ANNOTATIONS_DIR ADAPTED_VIDEO_ROOT CONFIG_DIR CONFIG_PATH CHECKPOINT EVAL_OUTPUT_DIR
export DOWNLOAD_DATA DOWNLOAD_PARTICIPANTS INCLUDE_PARTICIPANTS VAL_PARTICIPANTS LINK_METHOD
export VAL_RATIO HDEPIC_SPLIT_PRESET CONVERT_FPS
export EVAL_CPUS_PER_TASK EVAL_BATCH_SIZE EVAL_NUM_EPOCHS EVAL_NUM_WORKERS

if [[ "${MODE}" == "prep" ]]; then
    run_in_container "no_nv" "${prep_payload}"

    if [[ "${SUBMIT_EVAL}" == "1" ]]; then
        echo "[prep] Submitting dependent GPU eval job"
        eval_sbatch_args=(
            --parsable
            --dependency="afterok:${SLURM_JOB_ID}"
            --job-name=VJEPA2-EXP__hdepic_aa_eval
            --account=your_slurm_account
            --nodes=1
            --ntasks=1
            --gres="gpu:${EVAL_GPUS}"
            --cpus-per-task="${EVAL_CPUS_PER_TASK}"
            --mem="${EVAL_MEM}"
            --time="${EVAL_TIME}"
            --output="${PROJECT_ROOT}/logs/hdepic_aa_eval_%j.out"
            --error="${PROJECT_ROOT}/logs/hdepic_aa_eval_%j.err"
            --export=ALL,MODE=eval
        )
        if [[ -n "${EVAL_PARTITION}" ]]; then
            eval_sbatch_args+=(--partition="${EVAL_PARTITION}")
        fi
        if [[ -n "${EVAL_QOS}" ]]; then
            eval_sbatch_args+=(--qos="${EVAL_QOS}")
        fi
        eval_job_id=$(sbatch \
            "${eval_sbatch_args[@]}" \
            "$0")
        echo "[prep] Submitted eval job: ${eval_job_id}"
    else
        echo "[prep] SUBMIT_EVAL=${SUBMIT_EVAL}; not submitting eval job"
    fi
elif [[ "${MODE}" == "eval" ]]; then
    run_in_container "with_nv" "${eval_payload}"
else
    echo "Unknown MODE=${MODE}; expected prep or eval" >&2
    exit 1
fi

echo "========================================"
echo "End        : $(date)"
echo "========================================"
