#!/bin/bash
# Disjoint 256-row coarse target export. Held-out requires explicit unlock and
# must not be submitted until calibration target rules are frozen.
#SBATCH --job-name=b17_tpjepa_target_shard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=01:30:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"

DATA_ROOT=${DATA_ROOT}
CODE_ROOT="${PROJECT_ROOT}"
EXPECTED_CODE_ROOT=${PROJECT_ROOT}
export VJEPA_ROOT
PROTOCOL_ID=egtea-stream-mtp-pruning/task-protected-jepa-oracle-gate-l16-v1
TRAIN_CSV="$DATA_ROOT/data/egtea/vjepa_annotations/stream_half_split/split1/EGTEA_train_stream_mtp.csv"
VIDEO_ROOT="$DATA_ROOT/data/egtea/stream_session_videos"
CHECKPOINT="$DATA_ROOT/checkpoints/vitl.pt"
ENCODER_LORA="$DATA_ROOT/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt"
PREDICTOR_LORA="$DATA_ROOT/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt"
PARENT_CKPT="$DATA_ROOT/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt"
MANIFEST_SPLIT="${MANIFEST_SPLIT:-calibration}"
SHARD_START="${SHARD_START:?set SHARD_START to a registered 256-row boundary}"
SHARD_COUNT="${SHARD_COUNT:-256}"
ALLOW_HELDOUT="${ALLOW_HELDOUT:-0}"

[[ "$(realpath "$CODE_ROOT")" == "$EXPECTED_CODE_ROOT" ]] \
  || { echo "ERROR: wrong isolated code root" >&2; exit 2; }
[[ "$(realpath "$CODE_ROOT")" == "$(realpath "$PROJECT_ROOT")" ]] \
  || { echo "ERROR: wrong shared code root" >&2; exit 2; }
[[ "$SHARD_COUNT" == "256" ]] || { echo "ERROR: target shard count must be 256" >&2; exit 2; }
case "$MANIFEST_SPLIT" in
  calibration)
    MANIFEST_CSV="${B17_MANIFEST_DIR}/oracle_calibration_512.csv"
    MANIFEST_SHA256=e09fa32c7351f4ffe0c387b70c708e76c0cec125b9df0959e0d58abb2111b18b
    [[ "$SHARD_START" == "0" || "$SHARD_START" == "256" ]] \
      || { echo "ERROR: calibration start must be 0 or 256" >&2; exit 2; }
    ;;
  heldout)
    [[ "$ALLOW_HELDOUT" == "1" ]] \
      || { echo "ERROR: held-out export locked until calibration target rules freeze" >&2; exit 2; }
    MANIFEST_CSV="${B17_MANIFEST_DIR}/oracle_heldout_2048.csv"
    MANIFEST_SHA256=937f1e140fc0eeb3942804cebdece5ecef43e39202d92be18f3a5d59fcdc9d42
    case "$SHARD_START" in
      0|256|512|768|1024|1280|1536|1792) ;;
      *) echo "ERROR: held-out start must be a registered 256-row boundary" >&2; exit 2;;
    esac
    ;;
  *) echo "ERROR: MANIFEST_SPLIT must be calibration or heldout" >&2; exit 2;;
esac

for path in "$TRAIN_CSV" "$MANIFEST_CSV" "$CHECKPOINT" "$ENCODER_LORA" \
  "$PREDICTOR_LORA" "$PARENT_CKPT"; do
  [[ -f "$path" ]] || { echo "ERROR: missing required file $path" >&2; exit 2; }
done
[[ -d "$VIDEO_ROOT" ]] || { echo "ERROR: missing video root $VIDEO_ROOT" >&2; exit 2; }

OUT_ROOT="$DATA_ROOT/outputs/b17_token_pruning/task_protected_jepa/oracle_targets/$MANIFEST_SPLIT"
mkdir -p "$OUT_ROOT"
SHARD_STOP=$((SHARD_START + SHARD_COUNT))
RUN_TAG="b17-tpjepa-coarse-${MANIFEST_SPLIT}-${SHARD_START}-${SHARD_STOP}-${SLURM_JOB_ID}"
OUT_TARGET="$OUT_ROOT/${RUN_TAG}.pt"
MONITOR_LOG="$OUT_ROOT/${RUN_TAG}.gpu.csv"
[[ ! -e "$OUT_TARGET" && ! -e "${OUT_TARGET}.json" ]] \
  || { echo "ERROR: refusing to overwrite $OUT_TARGET" >&2; exit 2; }
CODE_COMMIT="$(git -C "$CODE_ROOT" rev-parse HEAD)"
CODE_SNAPSHOT_SHA256="$({
  sha256sum \
    "$CODE_ROOT/app/hdepic_lora_action_anticipation/task_protected_jepa_pruning.py" \
    "$CODE_ROOT/app/hdepic_lora_action_anticipation/export_task_protected_jepa_targets.py" \
    "$CODE_ROOT/scripts/egtea/run_b17_tpjepa_target_shard.slurm" \
    "${B17_MANIFEST_DIR}/oracle_manifest.json"
  git -C "$CODE_ROOT" status --porcelain=v1 --untracked-files=all
} | sha256sum | awk '{print $1}')"

export DATA_ROOT CODE_ROOT VJEPA_ROOT PROTOCOL_ID TRAIN_CSV MANIFEST_SPLIT MANIFEST_CSV MANIFEST_SHA256
export VIDEO_ROOT CHECKPOINT ENCODER_LORA PREDICTOR_LORA PARENT_CKPT OUT_TARGET
export SHARD_START SHARD_COUNT CODE_COMMIT CODE_SNAPSHOT_SHA256
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$CODE_ROOT/scripts/egtea/gpu_monitor.sh"
start_gpu_monitor "$MONITOR_LOG" 15
trap stop_gpu_monitor EXIT INT TERM

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python -m app.hdepic_lora_action_anticipation.export_task_protected_jepa_targets \
  --protocol-id "$PROTOCOL_ID" \
  --manifest-split "$MANIFEST_SPLIT" \
  --manifest-csv "$MANIFEST_CSV" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --train-csv "$TRAIN_CSV" \
  --video-root "$VIDEO_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --parent-checkpoint "$PARENT_CKPT" \
  --encoder-lora "$ENCODER_LORA" \
  --predictor-lora "$PREDICTOR_LORA" \
  --output "$OUT_TARGET" \
  --shard-start "$SHARD_START" --shard-count "$SHARD_COUNT" \
  --group-side 8 --deletion-batch-size 1 --num-workers 2 \
  --job-id "$SLURM_JOB_ID" \
  --code-commit "$CODE_COMMIT" \
  --code-snapshot-sha256 "$CODE_SNAPSHOT_SHA256"
EOF

stop_gpu_monitor
trap - EXIT INT TERM
export GPU_MONITOR_INTERVAL
report_gpu_monitor
echo "target=$OUT_TARGET"
echo "monitor=$MONITOR_LOG"
