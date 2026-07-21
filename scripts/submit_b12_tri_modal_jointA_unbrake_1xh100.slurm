#!/bin/bash
#SBATCH --job-name=tri_modal_jtA
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384GB
# Keep wall-clock under the 2h util-kill window. Chain via auto-resubmit.
#SBATCH --time=01:50:00
#SBATCH --signal=B:USR1@90
#SBATCH --output=/scratch/ll5914/logs/tri_modal_jtA_%j.out
#SBATCH --error=/scratch/ll5914/logs/tri_modal_jtA_%j.err

set -euo pipefail

# Tri-modal joint "A" (unbrake fusion), warm from tri-modal joint 40.85%:
#   Joint recipe hit 40.85% @ep1 then flat-lined. Diagnosis: double brake —
#   zero-init W_o AND gate bias=-4 (sigmoid≈0.018) with fusion LR 1e-5 → the
#   fusion term never escapes noise level in 5 epochs; early-stop fired first.
#
# Changes vs joint run:
#   1) fusion lr_mult 0.5 → 10  (fusion/aux LR = 10 × 2e-5 = 2e-4; heads stay 2e-5)
#   2) gate_bias_init -4 → -1   (softer identity start for cold parts)
#   3) train anticipation jitter restored [0.25,1.75]  (ep1-peak looked like
#      fixed-1.0s overfit); frame cache disabled (keys need fixed indices,
#      random horizons would also grow the cache unbounded)
#   4) patience 4 → 6, epochs 10 → 15 (give fusion time to open)
#
# Train: fusion + predictor LoRA + heads.  Freeze: encoder LoRA + pooler.

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/submit_b12_tri_modal_jointA_unbrake_1xh100.slurm"

ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split"
VIDEO_ROOT_REMOTE="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
HDEPIC_ANN_ROOT="/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"

JOINT_RUN="/scratch/ll5914/experiments/tri_modal_joint_from_p01video_jointv2/action_anticipation_frozen/tri-modal-joint-from-videov2-vitl16-256-10ep-1xh100"

OUTPUT_DIR="/scratch/ll5914/experiments/tri_modal_jointA_unbrake"
CONFIG_PATH="/scratch/ll5914/logs/tri_modal_jointA_unbrake_1xh100_config.yaml"
TAG="tri-modal-jointA-unbrake-vitl16-256-15ep-1xh100"
RUN_DIR="${OUTPUT_DIR}/action_anticipation_frozen/${TAG}"
DONE_FLAG="${RUN_DIR}/TRAINING_DONE"

GAZE_EXTRACT_ROOT="/scratch/ll5914/datasets/HD-EPIC/_gaze_extract"
GAZE_ROOT="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze"
GAZE_SYNC_ROOT="/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos"
POSE_SLAM_ROOT="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze/P01/SLAM/multi"
POSE_MAPPING_JSON="${POSE_SLAM_ROOT}/vrs_to_multi_slam.json"

NUM_EPOCHS=15
PATIENCE=6
mkdir -p /scratch/ll5914/logs "${OUTPUT_DIR}" "${RUN_DIR}"

if [[ -f "${DONE_FLAG}" ]]; then
  echo "Already done (${DONE_FLAG}); exiting."
  exit 0
fi

STAGE_ROOT="${STAGE_ROOT:-/dev/shm/${USER}/hdepic_p01_${SLURM_JOB_ID:-manual}}"
VIDEO_ROOT="${STAGE_ROOT}"
_RESUBMITTED=0
request_resubmit() {
  if [[ -f "${DONE_FLAG}" ]]; then
    return 0
  fi
  if [[ "${_RESUBMITTED}" == "1" ]]; then
    return 0
  fi
  _RESUBMITTED=1
  echo "Auto-resubmitting next chunk (util-kill / walltime / continue)..."
  sbatch "${SCRIPT_PATH}" || echo "WARN: sbatch resubmit failed" >&2
}
cleanup_stage() {
  if [[ -n "${STAGE_ROOT:-}" && -d "${STAGE_ROOT}" ]]; then
    echo "Cleaning staged videos: ${STAGE_ROOT}"
    rm -rf "${STAGE_ROOT}" || true
  fi
}
on_signal() {
  echo "Caught signal — requesting resubmit before exit @ $(date)"
  request_resubmit
  if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
  fi
}
trap cleanup_stage EXIT
trap on_signal USR1 TERM

# Warm-start chain: this run's latest → this run's best → tri-modal joint 40.85%.
WARM_SRC="tri_modal_joint_40p85"
PROBE_CKPT="${JOINT_RUN}/best.pt"
ENCODER_LORA_CKPT="${JOINT_RUN}/encoder_lora_best.pt"
PREDICTOR_LORA_CKPT="${JOINT_RUN}/predictor_lora_best.pt"
FUSION_CKPT="${JOINT_RUN}/tri_modal_fusion_best.pt"
RESUME=0
BEST_FLOOR="40.847"

if [[ -f "${RUN_DIR}/latest.pt" && -f "${RUN_DIR}/tri_modal_fusion_latest.pt" ]]; then
  WARM_SRC="run_latest_resume"
  PROBE_CKPT="${RUN_DIR}/latest.pt"
  ENCODER_LORA_CKPT="${RUN_DIR}/encoder_lora_latest.pt"
  PREDICTOR_LORA_CKPT="${RUN_DIR}/predictor_lora_latest.pt"
  FUSION_CKPT="${RUN_DIR}/tri_modal_fusion_latest.pt"
  RESUME=1
elif [[ -f "${RUN_DIR}/best.pt" && -f "${RUN_DIR}/tri_modal_fusion_best.pt" ]]; then
  WARM_SRC="run_best_fresh_opt"
  PROBE_CKPT="${RUN_DIR}/best.pt"
  ENCODER_LORA_CKPT="${RUN_DIR}/encoder_lora_best.pt"
  PREDICTOR_LORA_CKPT="${RUN_DIR}/predictor_lora_best.pt"
  FUSION_CKPT="${RUN_DIR}/tri_modal_fusion_best.pt"
  RESUME=0
fi

echo "========================================================"
echo "Tri-modal joint A (unbrake fusion: LR 2e-4, gate -1, jitter)"
echo "Job  : ${SLURM_JOB_ID:-manual}  Node: $(hostname)"
echo "Start: $(date)"
echo "WarmSrc: ${WARM_SRC}  resume_checkpoint=${RESUME}"
echo "Probe : ${PROBE_CKPT}"
echo "EncLoRA: ${ENCODER_LORA_CKPT} (FROZEN)"
echo "PredLoRA: ${PREDICTOR_LORA_CKPT} (TRAIN)"
echo "Heads : TRAIN  pooler: FROZEN  Fusion: ${FUSION_CKPT} (TRAIN @2e-4)"
echo "Frame cache: DISABLED (anticipation jitter)"
echo "Floor : ${BEST_FLOOR}  Target: gaze+pose concat 42.74%"
echo "Staging videos: ${VIDEO_ROOT_REMOTE}/P01 -> ${STAGE_ROOT}"
df -h /dev/shm || true
echo "========================================================"

mkdir -p "${STAGE_ROOT}/P01"
echo "Copying P01 videos into RAM disk (follow symlinks)..."
rsync -aL --info=stats2 "${VIDEO_ROOT_REMOTE}/P01/" "${STAGE_ROOT}/P01/"
echo "Stage done: $(du -sh "${STAGE_ROOT}" | awk '{print $1}') @ $(date)"

for f in "${PROBE_CKPT}" "${ENCODER_LORA_CKPT}" "${PREDICTOR_LORA_CKPT}" "${FUSION_CKPT}" "${POSE_MAPPING_JSON}" "${CHECKPOINT}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing required file: ${f}" >&2
    exit 1
  fi
done

export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TRI_MODAL_PREFETCH_DEPTH=4
export TRI_MODAL_AUX_WORKERS=4
export DATALOADER_PREFETCH_DEPTH=2
export TRI_MODAL_GAZE_MAP_CACHE=256
export GAZE_DECODER_CACHE_READER=1
export GAZE_DECODER_NUM_THREADS="${GAZE_DECODER_NUM_THREADS:-1}"
export BEST_METRIC_FLOOR="${BEST_FLOOR}"
# NOTE: TRI_MODAL_FRAME_CACHE intentionally NOT set (random horizons).
export MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))

"${PYTHON}" - <<PY
import sys, yaml
from pathlib import Path
sys.path.insert(0, '${PROJECT_ROOT}')
sys.path.insert(0, '${VJEPA_ROOT}')
template_path = Path('${VJEPA_ROOT}/configs/eval/vitl/ek100.yaml')
config_path   = Path('${CONFIG_PATH}')
resume = bool(int('${RESUME}'))
with template_path.open() as f:
    cfg = yaml.safe_load(f)
cfg['nodes'] = 1
cfg['tasks_per_node'] = 1
cfg['cpus_per_task'] = 32
cfg['tag']  = '${TAG}'
cfg['eval_name'] = 'app.hdepic_lora_action_anticipation'
cfg['folder'] = '${OUTPUT_DIR}'
cfg['resume_checkpoint'] = resume
cfg['val_only'] = False
data = cfg['experiment']['data']
data.update({
    'dataset': 'EK100', 'file_format': 1,
    'base_path': '${VIDEO_ROOT}',
    'dataset_train': '${ANN_DIR}/HD_EPIC_train_vjepa.csv',
    'dataset_val': '${ANN_DIR}/HD_EPIC_val_vjepa.csv',
    'num_workers': 16, 'val_num_workers': 4, 'clip_balanced': True,
    'prefetch_factor': 2,
    'pin_memory': False,
    'resolution': 256, 'frames_per_clip': 32, 'frames_per_second': 8,
    'anticipation_time_sec': [1.0, 1.0],
    # Jitter restored: regularizes against the ep1-peak overfit seen at fixed 1.0s.
    'train_anticipation_time_sec': [0.25, 1.75],
})
opt = cfg['experiment']['optimization']
opt.update({
    'batch_size': 16, 'num_epochs': ${NUM_EPOCHS}, 'use_bfloat16': True,
    'use_focal_loss': False, 'val_every_epochs': 1, 'best_metric': 'val-action-top5',
    'best_metric_floor': float('${BEST_FLOOR}'),
    'early_stopping_patience': ${PATIENCE}, 'grad_clip': 1.0,
    # Heads LR 2e-5; predictor LoRA lr_mult=0.5 → 1e-5; fusion lr_mult=10 → 2e-4.
    'multihead_kwargs': [{
        'weight_decay': 0.0001, 'final_weight_decay': 0.0001,
        'lr': 0.00002, 'start_lr': 0.0, 'final_lr': 0.0, 'warmup': 1,
    }],
})
cfg['model_kwargs']['checkpoint'] = '${CHECKPOINT}'
cfg['model_kwargs']['module_name'] = 'evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar'
cfg['experiment']['classifier'] = {'num_probe_blocks': 4, 'num_heads': 16}
fusion_block = {
    'fusion_type': 'projected_tri_modal_cross_attention',
    'use_gaze_branch': True,
    'use_imu_branch': True,
    'token_budget_ratio': [1.0, 0.5, 0.1],
    'gaze_token_ratio': 0.5,
    'imu_token_ratio': 0.1,
    'gaze_grid_size': 10,
    'imu_encoder_type': 'gru',
    'fusion_num_heads': 4,
    'fusion_num_layers': 1,
    'use_gated_residual': True,
    'gate_bias_init': -1.0,
    'imu_hidden_dim': 128,
    # Fusion absolute LR = lr_mult * heads_lr = 10 * 2e-5 = 2e-4.
    'lr_mult': 10.0,
    'weight_decay': 0.0001,
    'dropout': 0.0,
    'load_checkpoint_path': '${FUSION_CKPT}',
}
cfg['experiment']['lora'] = {
    'enabled': True, 'rank': 8, 'alpha': 16.0, 'dropout': 0.05,
    'probe_train_mode': 'full', 'train_heads': True, 'freeze_pooler': True,
    'pretrained_probe': '${PROBE_CKPT}',
    'load_probe_heads': True,
    'align_reference_metrics': True, 'val_metric_scope': 'native',
    'val_metric_aggregation': 'metric_wise_max',
    'class_space': 'train_only', 'temporal_sampling': 'phd_reference',
    'hdepic_reference': {
        'annotations_pkl': '${HDEPIC_ANN_ROOT}/HD_EPIC_Narrations.pkl',
        'verb_classes_csv': '${HDEPIC_ANN_ROOT}/HD_EPIC_verb_classes.csv',
        'noun_classes_csv': '${HDEPIC_ANN_ROOT}/HD_EPIC_noun_classes.csv',
    },
    'encoder_lora': {
        'enabled': True, 'rank': 8, 'alpha': 16.0, 'dropout': 0.05,
        'last_n_blocks': 0, 'lr_mult': 0.5, 'weight_decay': 0.0001,
        'activation_checkpointing': True,
        'target_suffixes': ['attn.qkv', 'attn.proj'],
        'warm_start_at_init': True,
        'freeze': True,
        'load_checkpoint_path': '${ENCODER_LORA_CKPT}',
        'checkpoint_path': '${RUN_DIR}/encoder_lora_latest.pt',
    },
    'predictor_lora': {
        'enabled': True, 'rank': 8, 'alpha': 16.0, 'dropout': 0.05,
        'last_n_blocks': 0, 'lr_mult': 0.5, 'weight_decay': 0.0001,
        'activation_checkpointing': True,
        'target_suffixes': ['attn.qkv', 'attn.proj'],
        'warm_start_at_init': True,
        'freeze': False,
        'load_checkpoint_path': '${PREDICTOR_LORA_CKPT}',
        'checkpoint_path': '${RUN_DIR}/predictor_lora_latest.pt',
    },
    'gaze': {
        'mode': 'projected_tri_modal_cross_attention',
        'gaze_root': '${GAZE_ROOT}',
        'extract_root': '${GAZE_EXTRACT_ROOT}',
        'sync_root': '${GAZE_SYNC_ROOT}',
        'binary_radius_px': 64,
        'disable_train_aug': True,
        'fusion': fusion_block,
        'pose': {
            'enabled': True,
            'slam_root': '${POSE_SLAM_ROOT}',
            'mapping_json': '${POSE_MAPPING_JSON}',
            'feature_set': 'pose_6d',
            'quality_min': 0.0,
            'history_sec': 0,
            'interframe_k_max': 64,
            'cache_sessions': True,
        },
    },
}
config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open('w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f'Config written: {config_path} (resume_ckpt={resume})')
PY

cd "${VJEPA_ROOT}"
set +e
"${PYTHON}" -m evals.main --fname "${CONFIG_PATH}" --devices cuda:0 &
TRAIN_PID=$!
wait "${TRAIN_PID}"
train_rc=$?
TRAIN_PID=""
set -e

echo "Training process exit_code=${train_rc} at $(date)"

"${PYTHON}" - <<PY
from pathlib import Path
run = Path("${RUN_DIR}")
done = run / "TRAINING_DONE"
latest = run / "latest.pt"
num_epochs = int("${NUM_EPOCHS}")
patience = int("${PATIENCE}")
if latest.is_file():
    import torch
    ck = torch.load(latest, map_location="cpu", weights_only=False)
    epoch = int(ck.get("epoch", -1) or -1)
    bad = int(ck.get("bad_validations", 0) or 0)
    print(f"latest epoch={epoch} bad_validations={bad}")
    if epoch >= num_epochs - 1:
        done.write_text(f"finished epoch={epoch}\n")
        print("TRAINING_DONE: reached final epoch")
    elif bad >= patience:
        done.write_text(f"early_stop epoch={epoch} bad={bad}\n")
        print("TRAINING_DONE: early stop")
print(f"NEED_RESUBMIT={0 if done.is_file() else 1}")
PY

if [[ ! -f "${DONE_FLAG}" ]]; then
  request_resubmit
else
  echo "Tri-modal joint A complete: $(date)"
fi

echo "Checkpoints: ${RUN_DIR}/{best.pt,tri_modal_fusion_best.pt,predictor_lora_best.pt}"
if [[ "${_RESUBMITTED}" == "1" || -f "${DONE_FLAG}" ]]; then
  exit 0
fi
exit ${train_rc}
