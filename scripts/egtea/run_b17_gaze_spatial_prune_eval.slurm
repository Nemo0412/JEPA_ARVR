#!/bin/bash
# B17: paired frozen evaluation of spatial selectors under the exact
# loss-aware-derived K=3840 temporal quota.  Historical attention K=4096 is an
# external reference and must not be relabeled as a matched-budget arm.
#
# Env: MODE={smoke|full}. The registered v1 matrix and isolated code root are
# fail-closed; create a new protocol instead of mutating scientific parameters.
#SBATCH --job-name=VJEPA2-EXP__b17_gaze_spatial_prune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=01:50:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"

DATA_ROOT="${DATA_ROOT:-${DATA_ROOT}}"
EXPECTED_CODE_ROOT="${PROJECT_ROOT}"
CODE_ROOT="${CODE_ROOT:-${EXPECTED_CODE_ROOT}}"
export VJEPA_ROOT
MODE="${MODE:-smoke}"
VARIANTS="uniform,calib,attention,gaze,gaze_shift"
POSITION_MODE="true_full"
BUDGET="${BUDGET:-3840}"
BATCH_SIZE="${BATCH_SIZE:-8}"

if [[ "${BUDGET}" != "3840" ]]; then
  echo "ERROR: B17 v1 is explicitly K=3840; got BUDGET=${BUDGET}" >&2
  exit 2
fi
if [[ "$(realpath "${CODE_ROOT}")" != "${EXPECTED_CODE_ROOT}" ]]; then
  echo "ERROR: B17 runs must use isolated CODE_ROOT=${EXPECTED_CODE_ROOT}; got ${CODE_ROOT}" >&2
  exit 2
fi
if [[ "$(realpath "$CODE_ROOT")" != "$(realpath "$PROJECT_ROOT")" ]]; then
  echo "ERROR: CODE_ROOT does not match the shared project root" >&2
  exit 2
fi

ANN_DIR="${ANN_DIR:-${DATA_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/data/egtea/stream_session_videos}"
GAZE_ROOT="${GAZE_ROOT:-${DATA_ROOT}/data/egtea/gaze_data/gaze_data/gaze_data}"
CHECKPOINT="${CHECKPOINT:-${DATA_ROOT}/checkpoints/vitl.pt}"
ENCODER_LORA="${ENCODER_LORA:-${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt}"
PREDICTOR_LORA="${PREDICTOR_LORA:-${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt}"
PARENT_CKPT="${PARENT_CKPT:-${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt}"
CALIBRATED_SPATIAL_MASK="${CALIBRATED_SPATIAL_MASK:-${DATA_ROOT}/outputs/egtea_stream_mtp_kvcache_prune/pattern/full/keep_prob_loss_aware.npy}"
CALIBRATION_SCORES="${CALIBRATION_SCORES:-${DATA_ROOT}/outputs/egtea_stream_mtp_kvcache_prune/calibration/anticipation-r0.6-fps8.scores.pt}"
TRAIN_CSV="${TRAIN_CSV:-${ANN_DIR}/EGTEA_train_stream_mtp.csv}"
VAL_CSV="${VAL_CSV:-${ANN_DIR}/EGTEA_val_stream_mtp.csv}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/outputs/b17_token_pruning/gaze_spatial}"

for path in "${CHECKPOINT}" "${ENCODER_LORA}" "${PREDICTOR_LORA}" "${PARENT_CKPT}" \
  "${CALIBRATED_SPATIAL_MASK}" "${CALIBRATION_SCORES}" \
  "${TRAIN_CSV}" "${VAL_CSV}"; do
  [[ -f "${path}" ]] || { echo "ERROR: missing required file: ${path}" >&2; exit 2; }
done
for path in "${VIDEO_ROOT}" "${GAZE_ROOT}"; do
  [[ -d "${path}" ]] || { echo "ERROR: missing required directory: ${path}" >&2; exit 2; }
done
[[ -f "${CODE_ROOT}/app/hdepic_lora_action_anticipation/eval_stream_mtp_gaze_spatial_prune.py" ]] \
  || { echo "ERROR: B17 evaluator missing from CODE_ROOT=${CODE_ROOT}" >&2; exit 2; }

mkdir -p "${OUT_ROOT}"
if [[ "${MODE}" == "smoke" ]]; then
  VAL_SUBSET_N="${VAL_SUBSET_N:-128}"
  WORKERS="${WORKERS:-8}"
elif [[ "${MODE}" == "full" ]]; then
  VAL_SUBSET_N="${VAL_SUBSET_N:-0}"
  WORKERS="${WORKERS:-16}"
else
  echo "ERROR: MODE must be smoke or full, got ${MODE}" >&2
  exit 2
fi

VARIANT_TAG="${VARIANTS//,/-}"
RUN_TAG="b17v2-la3840-${VARIANT_TAG}-${POSITION_MODE}-${MODE}"
OUT_JSON="${OUT_ROOT}/${RUN_TAG}-${SLURM_JOB_ID}.json"
MONITOR_LOG="${OUT_ROOT}/${RUN_TAG}_${SLURM_JOB_ID}.gpu.csv"
[[ ! -e "${OUT_JSON}" ]] || { echo "ERROR: refusing to overwrite ${OUT_JSON}" >&2; exit 2; }

CODE_COMMIT="$(git -C "${CODE_ROOT}" rev-parse HEAD)"
CODE_SNAPSHOT_SHA256="$({
  sha256sum \
    "${CODE_ROOT}/app/hdepic_lora_action_anticipation/eval_stream_mtp_gaze_spatial_prune.py" \
    "${CODE_ROOT}/app/hdepic_lora_action_anticipation/eval_stream_mtp_fixed_budget_prune.py" \
    "${CODE_ROOT}/app/hdepic_lora_action_anticipation/temporal_budget_pruning.py" \
    "${CODE_ROOT}/app/hdepic_lora_action_anticipation/gaze_spatial_pruning.py" \
    "${CODE_ROOT}/scripts/egtea/run_b17_gaze_spatial_prune_eval.slurm"
  git -C "${CODE_ROOT}" status --porcelain=v1 --untracked-files=all
} | sha256sum | awk '{print $1}')"

export DATA_ROOT CODE_ROOT VJEPA_ROOT MODE VARIANTS POSITION_MODE BUDGET BATCH_SIZE
export TRAIN_CSV VAL_CSV VIDEO_ROOT GAZE_ROOT CHECKPOINT ENCODER_LORA PREDICTOR_LORA PARENT_CKPT
export CALIBRATED_SPATIAL_MASK CALIBRATION_SCORES VAL_SUBSET_N WORKERS OUT_JSON CODE_COMMIT CODE_SNAPSHOT_SHA256
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

source "${CODE_ROOT}/scripts/egtea/gpu_monitor.sh"
start_gpu_monitor "${MONITOR_LOG}" 15
trap stop_gpu_monitor EXIT INT TERM

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python -m app.hdepic_lora_action_anticipation.eval_stream_mtp_gaze_spatial_prune \
  --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" \
  --video-root "$VIDEO_ROOT" --gaze-root "$GAZE_ROOT" \
  --checkpoint "$CHECKPOINT" --init-from-ckpt "$PARENT_CKPT" \
  --encoder-lora "$ENCODER_LORA" --predictor-lora "$PREDICTOR_LORA" \
  --calibrated-spatial-mask "$CALIBRATED_SPATIAL_MASK" \
  --calibration-scores "$CALIBRATION_SCORES" \
  --variants "$VARIANTS" --position-mode "$POSITION_MODE" --budget "$BUDGET" \
  --max-frames 80 --fps 8 --src-fps 8 --img-size 256 \
  --horizons-sec 2,4,6 --primary-horizon-sec 2 --anticipation-sec 2 \
  --batch-size "$BATCH_SIZE" --num-workers "$WORKERS" \
  --val-subset-n "$VAL_SUBSET_N" --val-subset-seed 17 \
  --max-val-batches 0 --audit-samples 8 \
  --job-id "$SLURM_JOB_ID" --code-root "$CODE_ROOT" \
  --code-commit "$CODE_COMMIT" --code-snapshot-sha256 "$CODE_SNAPSHOT_SHA256" \
  --out-json "$OUT_JSON"
EOF

stop_gpu_monitor
trap - EXIT INT TERM
export GPU_MONITOR_INTERVAL
report_gpu_monitor
awk -F, '
  NR > 2 { value=$3; gsub(/[[:space:]]/, "", value) }
  NR > 2 && value ~ /^[0-9]+([.][0-9]+)?$/ { total += value; count += 1 }
  END {
    if (count > 0) printf "steady_gpu_util_after_2_samples=%.2f%% samples=%d\n", total/count, count
  }
' "${MONITOR_LOG}"
echo "B17 gaze spatial prune ${MODE} completed: ${OUT_JSON}"
