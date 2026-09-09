#!/bin/bash
# Representative 1- or 8-row coarse target export. This launcher is smoke-only;
# scientific shards use the separate exact-range launcher.
#SBATCH --job-name=b17_tpjepa_target_smoke
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
MANIFEST_CSV="${B17_MANIFEST_DIR}/oracle_calibration_512.csv"
MANIFEST_SHA256=e09fa32c7351f4ffe0c387b70c708e76c0cec125b9df0959e0d58abb2111b18b
VIDEO_ROOT="$DATA_ROOT/data/egtea/stream_session_videos"
CHECKPOINT="$DATA_ROOT/checkpoints/vitl.pt"
ENCODER_LORA="$DATA_ROOT/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt"
PREDICTOR_LORA="$DATA_ROOT/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt"
PARENT_CKPT="$DATA_ROOT/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt"
OUT_ROOT="$DATA_ROOT/outputs/b17_token_pruning/task_protected_jepa/oracle_target_smoke"
SHARD_COUNT="${SHARD_COUNT:-1}"

[[ "$(realpath "$CODE_ROOT")" == "$EXPECTED_CODE_ROOT" ]] \
  || { echo "ERROR: wrong isolated code root" >&2; exit 2; }
[[ "$(realpath "$CODE_ROOT")" == "$(realpath "$PROJECT_ROOT")" ]] \
  || { echo "ERROR: wrong shared code root" >&2; exit 2; }
for path in "$TRAIN_CSV" "$MANIFEST_CSV" "$CHECKPOINT" "$ENCODER_LORA" \
  "$PREDICTOR_LORA" "$PARENT_CKPT"; do
  [[ -f "$path" ]] || { echo "ERROR: missing required file $path" >&2; exit 2; }
done
[[ -d "$VIDEO_ROOT" ]] || { echo "ERROR: missing video root $VIDEO_ROOT" >&2; exit 2; }
[[ "$SHARD_COUNT" == "1" || "$SHARD_COUNT" == "8" ]] \
  || { echo "ERROR: smoke SHARD_COUNT must be 1 or 8" >&2; exit 2; }
mkdir -p "$OUT_ROOT"

RUN_TAG="b17-tpjepa-coarse-calibration-target-smoke-n${SHARD_COUNT}-${SLURM_JOB_ID}"
OUT_TARGET="$OUT_ROOT/${RUN_TAG}.pt"
MONITOR_LOG="$OUT_ROOT/${RUN_TAG}.gpu.csv"
[[ ! -e "$OUT_TARGET" && ! -e "${OUT_TARGET}.json" ]] \
  || { echo "ERROR: refusing to overwrite $OUT_TARGET" >&2; exit 2; }
CODE_COMMIT="$(git -C "$CODE_ROOT" rev-parse HEAD)"
CODE_SNAPSHOT_SHA256="$({
  sha256sum \
    "$CODE_ROOT/app/hdepic_lora_action_anticipation/task_protected_jepa_pruning.py" \
    "$CODE_ROOT/app/hdepic_lora_action_anticipation/export_task_protected_jepa_targets.py" \
    "$CODE_ROOT/scripts/egtea/run_b17_tpjepa_target_smoke.slurm" \
    "${B17_MANIFEST_DIR}/oracle_manifest.json"
  git -C "$CODE_ROOT" status --porcelain=v1 --untracked-files=all
} | sha256sum | awk '{print $1}')"

export DATA_ROOT CODE_ROOT VJEPA_ROOT PROTOCOL_ID TRAIN_CSV MANIFEST_CSV MANIFEST_SHA256
export VIDEO_ROOT CHECKPOINT ENCODER_LORA PREDICTOR_LORA PARENT_CKPT OUT_TARGET
export CODE_COMMIT CODE_SNAPSHOT_SHA256 SHARD_COUNT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
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
  --manifest-split calibration \
  --manifest-csv "$MANIFEST_CSV" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --train-csv "$TRAIN_CSV" \
  --video-root "$VIDEO_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --parent-checkpoint "$PARENT_CKPT" \
  --encoder-lora "$ENCODER_LORA" \
  --predictor-lora "$PREDICTOR_LORA" \
  --output "$OUT_TARGET" \
  --shard-start 0 --shard-count "$SHARD_COUNT" \
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
