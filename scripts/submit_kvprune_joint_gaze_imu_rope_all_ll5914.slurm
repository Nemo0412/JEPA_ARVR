#!/bin/bash
#SBATCH --job-name=kv2gi
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/ll5914/logs/kvprune_gaze_imu_rope_all_%j.out
#SBATCH --error=/scratch/ll5914/logs/kvprune_gaze_imu_rope_all_%j.err
#SBATCH --signal=B:USR1@120

# RoPE all (2s, probe RoPE on every self-attn block, last-prediction prune)
# plus Gaze+IMU best (5ch adapter, IMU CA L=3, keep-aux, soft-gate).
# A100, H100, and H200. Two GPUs. Videos staged on the node.

set -euo pipefail
export TMPDIR=/scratch/ll5914/tmp
mkdir -p "$TMPDIR" /scratch/ll5914/logs

HORIZON=2
ROPE=1
ONLY_BLOCK0=0
ROPE_TIME=abs
NGPUS=2
EPOCHS=8
PATIENCE=0

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/submit_kvprune_joint_gaze_imu_rope_all_ll5914.slurm"
TRAIN_CSV="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/HD_EPIC_train_vjepa.csv"
VAL_CSV="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/HD_EPIC_val_vjepa.csv"
VIDEO_ROOT_REMOTE="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"
CA_RUN="/scratch/ll5914/experiments/concat_plus_cross_attn_v2/action_anticipation_frozen/concat-plus-ca-v2-L3-keepaux-softgate-vitl16-256-12ep-1xh100"
OUT_ROOT="/scratch/ll5914/experiments/kvprune_joint_gaze_imu"
RUN_DIR="${OUT_ROOT}/joint_2s_rope_all_gaze_imu_slotrope"
DONE_FLAG="${RUN_DIR}/DONE"

export VJEPA_ROOT PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${VJEPA_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRI_MODAL_GAZE_MAP_CACHE=24
export GAZE_DECODER_CACHE_READER=1
export GAZE_DECODER_NUM_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

STAGE_DIR="/state/partition1/${USER}/hdepic_vjepa_videos"
need_kb=$((25 * 1024 * 1024))
parent="$(dirname "${STAGE_DIR}")"
mkdir -p "${parent}"
avail_kb="$(df -Pk "${parent}" | awk 'NR==2{print $4}')"
if [[ -z "${avail_kb}" || "${avail_kb}" -lt "${need_kb}" ]]; then
  echo "ERROR: ${parent} has ${avail_kb:-0} KB free, need ${need_kb}" >&2
  exit 1
fi
mkdir -p "${STAGE_DIR}/P01"
remote_p01="${VIDEO_ROOT_REMOTE}/P01"
marker="${STAGE_DIR}/.p01_stage_ok"
lock="${STAGE_DIR}/.stage.lock"
echo "Staging P01 → ${STAGE_DIR} @ $(date)"
(
  flock -w 1800 9 || exit 1
  n_remote="$(find "${remote_p01}" -maxdepth 1 \( -type f -o -type l \) | wc -l)"
  n_local="$(find "${STAGE_DIR}/P01" -maxdepth 1 -type f 2>/dev/null | wc -l)"
  if [[ -f "${marker}" && "${n_local}" -ge "${n_remote}" && "${n_remote}" -gt 0 ]]; then
    echo "Stage hit: ${n_local}/${n_remote} files"
  else
    rsync -aL --info=stats1 "${remote_p01}/" "${STAGE_DIR}/P01/"
    find "${STAGE_DIR}/P01" -maxdepth 1 -type f | wc -l > "${marker}"
    echo "Stage done @ $(date)"
  fi
) 9>"${lock}"
VIDEO_ROOT="${STAGE_DIR}"
echo "VIDEO_ROOT=${VIDEO_ROOT}"

mkdir -p "${OUT_ROOT}"
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NAME:-}" ]]; then
  while read -r sib; do
    [[ -z "${sib}" || "${sib}" == "${SLURM_JOB_ID}" ]] && continue
    echo "Canceling sibling ${sib} (same name ${SLURM_JOB_NAME})"
    scancel --signal=KILL "${sib}" 2>/dev/null || true
  done < <(squeue -u "${USER}" -n "${SLURM_JOB_NAME}" -h -o '%i' 2>/dev/null || true)
fi

if [[ -f "${DONE_FLAG}" ]]; then
  echo "Already done (${DONE_FLAG}); exiting."
  exit 0
fi

_RESUBMITTED=0
TRAIN_PID=""

request_resubmit() {
  if [[ -f "${DONE_FLAG}" || "${_RESUBMITTED}" == "1" ]]; then
    return 0
  fi
  _RESUBMITTED=1
  echo "Auto-resubmitting after job ${SLURM_JOB_ID:-}..."
  dep=()
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    dep=(--dependency="afterany:${SLURM_JOB_ID}")
  fi
  local ok=0
  for spec in "a100_tandon|gpu:a100:2" "h100_tandon|gpu:h100:2" "h200_tandon|gpu:h200:2"; do
    IFS='|' read -r part gres <<< "$spec"
    out="$(sbatch --job-name=kv2gi \
      --account=torch_pr_111_tandon_advanced \
      --partition="$part" \
      --gres="$gres" \
      --mem=192G \
      --cpus-per-task=16 \
      --time=1-00:00:00 \
      "${dep[@]}" \
      "${SCRIPT_PATH}" 2>&1)" || true
    echo "$part => ${out}"
    if grep -q 'Submitted batch job' <<<"${out}"; then
      ok=1
    fi
  done
  if [[ "${ok}" != "1" ]]; then
    echo "WARN: sbatch resubmit failed on all partitions" >&2
  fi
}

on_signal() {
  echo "Caught signal — TERM train (ckpt) then resubmit @ $(date)"
  request_resubmit
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    for _ in $(seq 1 90); do
      kill -0 "${TRAIN_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${TRAIN_PID}" 2>/dev/null; then
      echo "WARN: train still alive after 90s — escalating KILL" >&2
      kill -KILL "${TRAIN_PID}" 2>/dev/null || true
    fi
    wait "${TRAIN_PID}" 2>/dev/null || true
    TRAIN_PID=""
  fi
}

trap on_signal USR1 TERM

echo "========================================================"
echo "kvprune joint Gaze+IMU + RoPE all  H=2s  NGPUS=2"
echo "Job=${SLURM_JOB_ID:-manual} partition=${SLURM_JOB_PARTITION:-} $(date)"
echo "host=$(hostname) OUT=${RUN_DIR}"
echo "init=${CHECKPOINT}"
echo "gaze encoder/adapter/fusion from ${CA_RUN}"
echo "probe RoPE all blocks, rope_time=abs, prune=last, horizon=2s"
echo "========================================================"
nvidia-smi -L || true

cd "$PROJECT_ROOT"
"$PYTHON" -m py_compile \
  app/hdepic_lora_action_anticipation/stream_kvcache_attn_prune.py \
  scripts/run_local_kvprune_joint_2s.py

export DECORD_THREADS=2

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NGPUS}" \
  scripts/run_local_kvprune_joint_2s.py \
  --train-csv "${TRAIN_CSV}" \
  --val-csv "${VAL_CSV}" \
  --video-root "${VIDEO_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --out-dir "${OUT_ROOT}" \
  --horizon "${HORIZON}" \
  --rope "${ROPE}" \
  --rope-time "${ROPE_TIME}" \
  --only-block0 "${ONLY_BLOCK0}" \
  --gaze-imu 1 \
  --encoder-lora "${CA_RUN}/encoder_lora_best.pt" \
  --adapter-ckpt "${CA_RUN}/binary_input_adapter_best.pt" \
  --fusion-ckpt "${CA_RUN}/tri_modal_fusion_best.pt" \
  --gaze-root /scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze \
  --gaze-extract-root /scratch/ll5914/datasets/HD-EPIC/_gaze_extract \
  --gaze-sync-root /scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos \
  --pose-slam-root /scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze/P01/SLAM/multi \
  --pose-mapping-json /scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze/P01/SLAM/multi/vrs_to_multi_slam.json \
  --adapter-lr-mult 0.25 \
  --fusion-lr-mult 2.0 \
  --resume 1 \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size 1 \
  --num-workers 6 \
  --val-num-workers 6 \
  --lr 2e-5 \
  --lora-lr-mult 0.5 \
  --ckpt-every 2000 \
  &
TRAIN_PID=$!
set +e
wait "${TRAIN_PID}"
train_rc=$?
set -e
TRAIN_PID=""

echo "Training exit_code=${train_rc} at $(date)"
if [[ -f "${DONE_FLAG}" ]]; then
  echo "complete: $(date)"
  exit 0
fi
request_resubmit
exit "${train_rc}"
