#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT}
export CODE_ROOT="${PROJECT_ROOT}"
export VJEPA_ROOT
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PROJECT_ROOT="$CODE_ROOT" SHARED_PROJECT_ROOT="$DATA_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python -m py_compile app/hdepic_lora_action_anticipation/probe_b18_q3_boundaries.py
python -m app.hdepic_lora_action_anticipation.probe_b18_q3_boundaries prepare \
 --train-csv "${DATA_ROOT}/data/egtea/vjepa_annotations/stream_16s_split/split1/EGTEA_train_stream_mtp.csv" \
 --video-root "${DATA_ROOT}/data/egtea/stream_session_videos" \
 --checkpoint "${DATA_ROOT}/checkpoints/vitl.pt" \
 --parent "${DATA_ROOT}/outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt" \
 --encoder-lora "${DATA_ROOT}/outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt" \
 --predictor-lora "${DATA_ROOT}/outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt" \
 --out "${DATA_ROOT}/outputs/attn_corner_sink/q3_boundaries/prepared" --workers 4
EOF
