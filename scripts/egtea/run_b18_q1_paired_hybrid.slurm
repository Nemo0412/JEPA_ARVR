#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q1_paired_hybrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=00:30:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT}
export CODE_ROOT="${PROJECT_ROOT}"
export VJEPA_ROOT
export Q1_ROOT="${DATA_ROOT}/outputs/attn_corner_sink/q1_paired_hybrid"
export Q1_START="${Q1_START:-0}" Q1_STOP="${Q1_STOP:-32}" Q1_BATCH="${Q1_BATCH:-4}" Q1_WORKERS="${Q1_WORKERS:-4}"
export Q1_TAG="${Q1_TAG:-b18-q1-paired-hybrid-smoke}"
export Q1_OUT="${Q1_ROOT}/${Q1_TAG}-${SLURM_JOB_ID}"
export Q1_PARITY="${Q1_PARITY:-1}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$Q1_OUT"
GPU_MONITOR_LOG="${Q1_OUT}/gpu.csv"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "${Q1_OUT}/gpu_identity.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits --loop=5 > "$GPU_MONITOR_LOG" &
GPU_MONITOR_PID=$!
stop_monitor() { kill "$GPU_MONITOR_PID" 2>/dev/null || true; wait "$GPU_MONITOR_PID" 2>/dev/null || true; }
trap stop_monitor EXIT INT TERM
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
ann="${DATA_ROOT}/data/egtea/vjepa_annotations/stream_16s_split/split1"
extra=()
[[ "$Q1_PARITY" == 1 ]] && extra+=(--parity-check)
python -m app.hdepic_lora_action_anticipation.eval_b18_q1_paired_hybrid \
 --train-csv "${ann}/EGTEA_train_stream_mtp.csv" --val-csv "${ann}/EGTEA_val_stream_mtp.csv" \
 --video-root "${DATA_ROOT}/data/egtea/stream_session_videos" \
 --checkpoint "${DATA_ROOT}/checkpoints/vitl.pt" \
 --init-from-ckpt "${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt" \
 --encoder-lora "${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt" \
 --predictor-lora "${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt" \
 --calib-l0 "${DATA_ROOT}/outputs/attn_corner_sink/pred_offline_calib/calib_predblk0_map_64x256.npy" \
 --calib-l11 "${DATA_ROOT}/outputs/attn_corner_sink/pred_offline_calib/calib_predblk11_map_64x256.npy" \
 --manifest "${DATA_ROOT}/outputs/attn_corner_sink/q1_evidence_audit_20260905/paired4000_manifest.csv" --n-samples 4000 --seed 20260905 \
 --start "$Q1_START" --stop "$Q1_STOP" --batch-size "$Q1_BATCH" --workers "$Q1_WORKERS" \
 --tag "$Q1_TAG" --out-dir "$Q1_OUT" "${extra[@]}"
EOF
