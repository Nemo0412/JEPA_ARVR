#!/bin/bash
#SBATCH --job-name=kvprune2
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/ll5914/logs/kvprune_joint_%j.out
#SBATCH --error=/scratch/ll5914/logs/kvprune_joint_%j.err
#SBATCH --signal=B:USR1@120

set -euo pipefail
export TMPDIR=/scratch/ll5914/tmp
mkdir -p "$TMPDIR" /scratch/ll5914/logs

# Stream KV + probe-blk0 prune; joint encoder-LoRA + probe from vitl.pt scratch.
# Protocol: fill 128 → prune drop 34 / keep 94 → encode new 34 (~every 4s) → probe.
# Vanilla RGB (no gaze/IMU). No predictor.
#
# Race (2 GPUs per arm → 4 GPUs for norope+rope):
#   for H in 2 6; do
#     for R in 0 1; do
#       name=kvprune${H}$([ "$R" = 1 ] && echo r || echo n)
#       for spec in 'a100_tandon|gpu:a100:2' 'h100_tandon|gpu:h100:2' 'h200_tandon|gpu:h200:2'; do
#         IFS='|' read -r part gres <<< "$spec"
#         sbatch --job-name="$name" --partition="$part" --gres="$gres" \
#           --mem=192G --cpus-per-task=16 --time=1-00:00:00 \
#           --export=HORIZON=$H,ROPE=$R,NGPUS=2 this.slurm
#       done
#     done
#   done

HORIZON="${HORIZON:-2}"
ROPE="${ROPE:-0}"
ONLY_BLOCK0="${ONLY_BLOCK0:-0}"
# 1 = no stream KV; encode FRAMES with full attention and pass every token to the probe.
NO_KV="${NO_KV:-0}"
FRAMES="${FRAMES:-32}"
# 1 = new 34 frames self-attend only, then concat with the kept 94 for the probe.
# Default 0 keeps the original forward (new frames attend to the kept K/V).
NEW_SELF_ONLY="${NEW_SELF_ONLY:-0}"
# GPUs per job (each arm = one sbatch). Default 1 for easier queueing.
NGPUS="${NGPUS:-1}"
case "${HORIZON}" in 2|6) ;; *) echo "ERROR: HORIZON must be 2 or 6" >&2; exit 2 ;; esac
case "${ROPE}" in 0|1) ;; *) echo "ERROR: ROPE must be 0 or 1" >&2; exit 2 ;; esac
case "${NGPUS}" in 1|2|4|8) ;; *) echo "ERROR: NGPUS must be 1/2/4/8" >&2; exit 2 ;; esac
case "${NO_KV}" in 0|1) ;; *) echo "ERROR: NO_KV must be 0 or 1" >&2; exit 2 ;; esac
case "${NEW_SELF_ONLY}" in 0|1) ;; *) echo "ERROR: NEW_SELF_ONLY must be 0 or 1" >&2; exit 2 ;; esac
if [[ "${NO_KV}" == "1" && "${NEW_SELF_ONLY}" == "1" ]]; then
  echo "ERROR: NEW_SELF_ONLY is the KV path" >&2
  exit 2
fi
if [[ "${NO_KV}" == "1" ]]; then
  ROPE=0
  if [[ "${FRAMES}" -le 0 || $((FRAMES % 2)) -ne 0 ]]; then
    echo "ERROR: FRAMES must be a positive multiple of 2 when NO_KV=1" >&2
    exit 2
  fi
fi

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/submit_kvprune_joint_ll5914.slurm"
TRAIN_CSV="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/HD_EPIC_train_vjepa.csv"
VAL_CSV="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/HD_EPIC_val_vjepa.csv"
VIDEO_ROOT_REMOTE="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"

export VJEPA_ROOT PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${VJEPA_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Stage P01 (~19GB) to node-local disk (scratch decode was ~20s/fetch).
STAGE_LOCAL="${STAGE_LOCAL:-1}"
VIDEO_ROOT="${VIDEO_ROOT_REMOTE}"
if [[ "${STAGE_LOCAL}" == "1" ]]; then
  remote_p01="${VIDEO_ROOT_REMOTE}/P01"
  STAGE_DIR=""
  need_kb=$((25 * 1024 * 1024))
  for cand in \
      "/state/partition1/${USER}/hdepic_vjepa_videos" \
      "/dev/shm/${USER}/hdepic_vjepa_videos"; do
    parent="$(dirname "${cand}")"
    mkdir -p "${parent}" 2>/dev/null || continue
    avail_kb="$(df -Pk "${parent}" 2>/dev/null | awk 'NR==2{print $4}')"
    [[ -n "${avail_kb}" && "${avail_kb}" -ge "${need_kb}" ]] || continue
    STAGE_DIR="${cand}"
    break
  done
  if [[ -n "${STAGE_DIR}" && -d "${remote_p01}" ]]; then
    mkdir -p "${STAGE_DIR}/P01"
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
    ) 9>"${lock}" && VIDEO_ROOT="${STAGE_DIR}" || echo "WARN: stage failed; using scratch"
  else
    echo "WARN: no local stage target; using scratch"
  fi
fi
echo "VIDEO_ROOT=${VIDEO_ROOT}"

ROPE_TIME="${ROPE_TIME:-abs}"
if [[ "${NO_KV}" == "1" ]]; then
  # Run root is the experiment dir. No joint_ prefix: this is a plain clip encode.
  OUT_ROOT="/scratch/ll5914/experiments/nokv_f${FRAMES}_h${HORIZON}s"
  ROPE_TAG=""
elif [[ "${NEW_SELF_ONLY}" == "1" ]]; then
  OUT_ROOT="/scratch/ll5914/experiments/kv_self34_h${HORIZON}s"
  ROPE_TAG=""
else
  OUT_ROOT="/scratch/ll5914/experiments/kvprune_joint_h${HORIZON}s"
  ROPE_TAG=$([ "${ROPE}" = "1" ] && echo rope_all || echo norope)
  if [[ "${ROPE}" = "1" && "${ROPE_TIME}" == "relpred" ]]; then
    ROPE_TAG="${ROPE_TAG}_relpred"
  fi
fi
if [[ "${NO_KV}" == "1" || "${NEW_SELF_ONLY}" == "1" ]]; then
  RUN_DIR="${OUT_ROOT}"
else
  RUN_DIR="${OUT_ROOT}/joint_${HORIZON}s_${ROPE_TAG}"
fi
DONE_FLAG="${RUN_DIR}/DONE"
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
  echo "Auto-resubmitting HORIZON=${HORIZON} ROPE=${ROPE} after job ${SLURM_JOB_ID:-}..."
  dep=()
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    dep=(--dependency="afterany:${SLURM_JOB_ID}")
  fi
  export TMPDIR="${TMPDIR:-/scratch/ll5914/tmp}"
  mkdir -p "${TMPDIR}"
  local mem_gb="${SLURM_MEM_PER_NODE:-96G}"
  if [[ "${mem_gb}" =~ ^[0-9]+$ ]]; then
    mem_gb="$((mem_gb / 1024))G"
  fi
  local ng="${NGPUS:-1}"
  # Scale mem with GPU count when continuing from a 1-GPU allocation.
  if [[ "${ng}" -ge 2 && "${mem_gb}" == "96G" ]]; then
    mem_gb="192G"
  fi
  local ok=0
  # 8 train/val DataLoader workers need headroom (was often stuck at 8 on 1-GPU).
  local cpus="${SLURM_CPUS_PER_TASK:-$((8 * ng))}"
  local cpus_min=$((8 * ng))
  if [[ "${cpus_min}" -lt 16 ]]; then cpus_min=16; fi
  if [[ "${cpus}" -lt "${cpus_min}" ]]; then
    cpus="${cpus_min}"
  fi
  # Prefer same GPU count on each partition.
  for spec in "h100_tandon|gpu:h100:${ng}" "h200_tandon|gpu:h200:${ng}"; do
    IFS='|' read -r part gres <<< "$spec"
    out="$(sbatch --job-name="${SLURM_JOB_NAME:-kvprune}" \
      --account="${SLURM_JOB_ACCOUNT:-torch_pr_111_tandon_advanced}" \
      --partition="$part" \
      --gres="$gres" \
      --mem="${mem_gb}" \
      --cpus-per-task="${cpus}" \
      --time="${SLURM_TIMELIMIT:-1-00:00:00}" \
      "${dep[@]}" \
      --export=HORIZON="${HORIZON}",ROPE="${ROPE}",ONLY_BLOCK0="${ONLY_BLOCK0}",ROPE_TIME="${ROPE_TIME:-abs}",NO_KV="${NO_KV:-0}",FRAMES="${FRAMES:-32}",NEW_SELF_ONLY="${NEW_SELF_ONLY:-0}",STAGE_LOCAL=1,NGPUS="${ng}",CKPT_EVERY="${CKPT_EVERY:-2000}",EPOCHS="${EPOCHS:-10}",PATIENCE="${PATIENCE:-0}" \
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
  # USR1 arrives ~120s before wall. Python handles TERM → compact latest.pt then exit 75.
  echo "Caught signal — TERM train (ckpt) then resubmit @ $(date)"
  request_resubmit
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    # Give torchrun/worker time to flush latest.pt before job hard-kills.
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
echo "kvprune joint  H=${HORIZON}s  ROPE=${ROPE} only_block0=${ONLY_BLOCK0} NO_KV=${NO_KV} FRAMES=${FRAMES} NEW_SELF_ONLY=${NEW_SELF_ONLY} NGPUS=${NGPUS}"
echo "Job=${SLURM_JOB_ID:-manual} partition=${SLURM_JOB_PARTITION:-} $(date)"
echo "host=$(hostname) OUT=${RUN_DIR}"
echo "init=${CHECKPOINT}  (probe+LoRA scratch / resume latest.pt)"
if [[ "${NO_KV}" == "1" ]]; then
  echo "no KV cache; full-attention encode ${FRAMES} frames → probe; Probe temporal RoPE OFF"
elif [[ "${NEW_SELF_ONLY}" == "1" ]]; then
  echo "new 34 frames encoder self-attn only; concat kept 94 → probe; Probe RoPE all"
else
  echo "cache=128f new=34f keep=94; new frames attend kept K/V; scores=Probe.blocks[0] detached"
fi
echo "========================================================"
nvidia-smi -L || true

cd "$PROJECT_ROOT"
"$PYTHON" -m py_compile \
  app/hdepic_lora_action_anticipation/stream_kvcache_attn_prune.py \
  scripts/run_local_kvprune_joint_2s.py

# 16 CPUs shared by all ranks. One thread per worker left the GPU waiting on
# MP4 decode even after videos were staged locally. Give each decoder 2 threads
# and keep total threads within the CPU allocation.
export DECORD_THREADS="${DECORD_THREADS:-2}"
_cpus="${SLURM_CPUS_PER_TASK:-16}"
_nw=$(( _cpus / DECORD_THREADS / NGPUS ))
if (( _nw < 2 )); then _nw=2; fi
if (( _nw > 8 )); then _nw=8; fi
echo "dataloader workers/rank=${_nw} DECORD_THREADS=${DECORD_THREADS} cpus=${_cpus}"

# Multi-GPU DDP via torchrun (NGPUS from sbatch --gres / --export).
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NGPUS}" \
  scripts/run_local_kvprune_joint_2s.py \
  --train-csv "${TRAIN_CSV}" \
  --val-csv "${VAL_CSV}" \
  --video-root "${VIDEO_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --out-dir "${OUT_ROOT}" \
  --horizon "${HORIZON}" \
  --rope "${ROPE}" \
  --rope-time "${ROPE_TIME:-abs}" \
  --only-block0 "${ONLY_BLOCK0}" \
  --no-kv "${NO_KV}" \
  --frames "${FRAMES}" \
  --new-self-only "${NEW_SELF_ONLY}" \
  --resume 1 \
  --epochs "${EPOCHS:-10}" \
  --patience "${PATIENCE:-0}" \
  --batch-size 1 \
  --num-workers "${_nw}" \
  --val-num-workers "${_nw}" \
  --lr 2e-5 \
  --lora-lr-mult 0.5 \
  --ckpt-every "${CKPT_EVERY:-2000}" \
  &
TRAIN_PID=$!
set +e
wait "${TRAIN_PID}"
train_rc=$?
set -e
TRAIN_PID=""

echo "Training exit_code=${train_rc} H=${HORIZON} ROPE=${ROPE} at $(date)"
if [[ -f "${DONE_FLAG}" ]]; then
  echo "complete: $(date)"
  exit 0
fi
# 75 = graceful wall/signal exit after writing latest.pt (see train script).
if [[ "${train_rc}" -eq 0 ]]; then
  # Script writes DONE on success; if missing, treat as incomplete.
  echo "WARN: exit 0 but no DONE flag — resubmitting"
fi
request_resubmit
exit "${train_rc}"
