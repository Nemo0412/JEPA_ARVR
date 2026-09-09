#!/bin/bash
# B17 v2: bounded same-budget attention/JEPA group-swap oracle.
# Set SHARD_COUNT=2 or 8 only for monitored throughput smoke; the scientific
# run is frozen to SHARD_COUNT=128 over the leading held-out-train manifest rows.
#SBATCH --job-name=VJEPA2-EXP__b17_tpjepa_swap_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=01:50:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"

DATA_ROOT=${DATA_ROOT}
CODE_ROOT="${PROJECT_ROOT}"
EXPECTED_CODE_ROOT=${PROJECT_ROOT}
export VJEPA_ROOT
PROTOCOL_ID=egtea-stream-mtp-pruning/attention-jepa-disagreement-swap-oracle-l16-v2
TRAIN_CSV="$DATA_ROOT/data/egtea/vjepa_annotations/stream_half_split/split1/EGTEA_train_stream_mtp.csv"
MANIFEST_CSV="${B17_MANIFEST_DIR}/oracle_heldout_2048.csv"
MANIFEST_SHA256=937f1e140fc0eeb3942804cebdece5ecef43e39202d92be18f3a5d59fcdc9d42
VIDEO_ROOT="$DATA_ROOT/data/egtea/stream_session_videos"
CHECKPOINT="$DATA_ROOT/checkpoints/vitl.pt"
ENCODER_LORA="$DATA_ROOT/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt"
PREDICTOR_LORA="$DATA_ROOT/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt"
PARENT_CKPT="$DATA_ROOT/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt"
SHARD_COUNT="${SHARD_COUNT:-128}"

[[ "$(realpath "$CODE_ROOT")" == "$EXPECTED_CODE_ROOT" ]] \
  || { echo "ERROR: wrong isolated code root" >&2; exit 2; }
[[ "$(realpath "$CODE_ROOT")" == "$(realpath "$PROJECT_ROOT")" ]] \
  || { echo "ERROR: wrong shared code root" >&2; exit 2; }
case "$SHARD_COUNT" in 2|8|128) ;; *) echo "ERROR: SHARD_COUNT must be 2, 8, or 128" >&2; exit 2;; esac
for path in "$TRAIN_CSV" "$MANIFEST_CSV" "$CHECKPOINT" "$ENCODER_LORA" \
  "$PREDICTOR_LORA" "$PARENT_CKPT"; do
  [[ -f "$path" ]] || { echo "ERROR: missing required file $path" >&2; exit 2; }
done
[[ -d "$VIDEO_ROOT" ]] || { echo "ERROR: missing video root $VIDEO_ROOT" >&2; exit 2; }

OUT_ROOT="$DATA_ROOT/outputs/b17_token_pruning/task_protected_jepa/attention_jepa_swap_v2"
mkdir -p "$OUT_ROOT"
if [[ "$SHARD_COUNT" == "128" ]]; then
  RUN_KIND=scientific
else
  RUN_KIND=smoke
fi
RUN_TAG="b17-tpjepa-swap-v2-${RUN_KIND}-n${SHARD_COUNT}-${SLURM_JOB_ID}"
OUT_ARTIFACT="$OUT_ROOT/${RUN_TAG}.pt"
MONITOR_LOG="$OUT_ROOT/${RUN_TAG}.gpu.csv"
[[ ! -e "$OUT_ARTIFACT" && ! -e "${OUT_ARTIFACT}.json" ]] \
  || { echo "ERROR: refusing to overwrite $OUT_ARTIFACT" >&2; exit 2; }
CODE_COMMIT="$(git -C "$CODE_ROOT" rev-parse HEAD)"
CODE_SNAPSHOT_SHA256="$({
  sha256sum \
    "$CODE_ROOT/app/hdepic_lora_action_anticipation/task_protected_jepa_pruning.py" \
    "$CODE_ROOT/app/hdepic_lora_action_anticipation/evaluate_attention_jepa_swap_oracle.py" \
    "$CODE_ROOT/scripts/analyze_b17_tpjepa_swap_oracle.py" \
    "$CODE_ROOT/scripts/egtea/run_b17_tpjepa_swap_oracle.slurm" \
    "$CODE_ROOT/configs/evaluation_protocols/egtea_stream_mtp_pruning.yaml"
  git -C "$CODE_ROOT" status --porcelain=v1 --untracked-files=all
} | sha256sum | awk '{print $1}')"

export DATA_ROOT CODE_ROOT VJEPA_ROOT PROTOCOL_ID TRAIN_CSV MANIFEST_CSV MANIFEST_SHA256
export VIDEO_ROOT CHECKPOINT ENCODER_LORA PREDICTOR_LORA PARENT_CKPT OUT_ARTIFACT
export CODE_COMMIT CODE_SNAPSHOT_SHA256 SHARD_COUNT PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$CODE_ROOT/scripts/egtea/gpu_monitor.sh"
start_gpu_monitor "$MONITOR_LOG" 15
trap stop_gpu_monitor EXIT INT TERM

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python -m app.hdepic_lora_action_anticipation.evaluate_attention_jepa_swap_oracle \
  --protocol-id "$PROTOCOL_ID" \
  --manifest-csv "$MANIFEST_CSV" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --train-csv "$TRAIN_CSV" \
  --video-root "$VIDEO_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --parent-checkpoint "$PARENT_CKPT" \
  --encoder-lora "$ENCODER_LORA" \
  --predictor-lora "$PREDICTOR_LORA" \
  --output "$OUT_ARTIFACT" \
  --shard-start 0 --shard-count "$SHARD_COUNT" --num-workers 2 \
  --job-id "$SLURM_JOB_ID" \
  --code-commit "$CODE_COMMIT" \
  --code-snapshot-sha256 "$CODE_SNAPSHOT_SHA256"
EOF

stop_gpu_monitor
trap - EXIT INT TERM
export GPU_MONITOR_INTERVAL
report_gpu_monitor
echo "artifact=$OUT_ARTIFACT"
echo "monitor=$MONITOR_LOG"
