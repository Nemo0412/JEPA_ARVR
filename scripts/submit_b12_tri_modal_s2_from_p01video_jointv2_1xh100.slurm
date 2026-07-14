#!/bin/bash
#SBATCH --job-name=tri_modal_s2
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384GB
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ll5914/logs/tri_modal_s2_%j.out
#SBATCH --error=/scratch/ll5914/logs/tri_modal_s2_%j.err

set -euo pipefail

# Tri-modal Stage-2 warm-started from Video joint v2 (heads) best Top-5=40.44% @ep3.
#
# Warm-start:
#   - probe heads from joint-v2 best.pt  (pooler stays frozen)
#   - encoder LoRA from joint-v2 encoder_lora_best.pt  (continue train)
#   - predictor LoRA from joint-v2 predictor_lora_best.pt  (FROZEN)
# New train: GazeSpatialEncoder + ImuTemporalEncoder + ProjectedTriModalCrossAttention
# Soft FT: classifier heads @ 2e-5; fusion lr_mult=0.25 → ~5e-6

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"

ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split"
VIDEO_ROOT="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
HDEPIC_ANN_ROOT="/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"

JOINT_V2_RUN="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep"
PROBE_CKPT="${JOINT_V2_RUN}/best.pt"
ENCODER_LORA_CKPT="${JOINT_V2_RUN}/encoder_lora_best.pt"
PREDICTOR_LORA_CKPT="${JOINT_V2_RUN}/predictor_lora_best.pt"

OUTPUT_DIR="/scratch/ll5914/experiments/tri_modal_fusion_s2_from_p01video_jointv2"
CONFIG_PATH="/scratch/ll5914/logs/tri_modal_fusion_s2_from_p01video_jointv2_1xh100_config.yaml"
TAG="tri-modal-s2-from-p01video-jointv2-ep3best-vitl16-256-10ep-1xh100"
RUN_DIR="${OUTPUT_DIR}/action_anticipation_frozen/${TAG}"

GAZE_EXTRACT_ROOT="/scratch/ll5914/datasets/HD-EPIC/_gaze_extract"
GAZE_ROOT="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze"
GAZE_SYNC_ROOT="/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos"
POSE_SLAM_ROOT="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze/P01/SLAM/multi"
POSE_MAPPING_JSON="${POSE_SLAM_ROOT}/vrs_to_multi_slam.json"

mkdir -p /scratch/ll5914/logs "${OUTPUT_DIR}"

RESUME=0
if [[ -f "${RUN_DIR}/latest.pt" && -f "${RUN_DIR}/tri_modal_fusion_latest.pt" ]]; then
  RESUME=1
  PROBE_CKPT="${RUN_DIR}/latest.pt"
  if [[ -f "${RUN_DIR}/encoder_lora_latest.pt" ]]; then
    ENCODER_LORA_CKPT="${RUN_DIR}/encoder_lora_latest.pt"
  fi
  if [[ -f "${RUN_DIR}/predictor_lora_latest.pt" ]]; then
    PREDICTOR_LORA_CKPT="${RUN_DIR}/predictor_lora_latest.pt"
  fi
fi

echo "========================================================"
echo "Tri-modal Stage-2 from Video joint v2 (40.44% @ep3) — 1xH100"
echo "Job  : ${SLURM_JOB_ID:-manual}  Node: $(hostname)"
echo "Start: $(date)"
echo "Resume: ${RESUME}"
echo "Probe : ${PROBE_CKPT}"
echo "EncLoRA: ${ENCODER_LORA_CKPT}"
echo "PredLoRA: ${PREDICTOR_LORA_CKPT} (frozen)"
echo "========================================================"

for f in "${PROBE_CKPT}" "${ENCODER_LORA_CKPT}" "${PREDICTOR_LORA_CKPT}" "${POSE_MAPPING_JSON}" "${CHECKPOINT}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing required file: ${f}" >&2
    exit 1
  fi
done

export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TRI_MODAL_PREFETCH_DEPTH=4
export GAZE_DECODER_CACHE_READER=1

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
    'num_workers': 16, 'val_num_workers': 8, 'clip_balanced': True,
    'prefetch_factor': 8,
    'resolution': 256, 'frames_per_clip': 32, 'frames_per_second': 8,
    'anticipation_time_sec': [1.0, 1.0],
    'train_anticipation_time_sec': [0.25, 1.75],
})
opt = cfg['experiment']['optimization']
opt.update({
    'batch_size': 16, 'num_epochs': 10, 'use_bfloat16': True,
    'use_focal_loss': False, 'val_every_epochs': 3, 'best_metric': 'val-action-top5',
    'early_stopping_patience': 3, 'grad_clip': 1.0,
    # Match joint-v2 head LR so we don't wash out the 40.44% heads.
    'multihead_kwargs': [{
        'weight_decay': 0.0001, 'final_weight_decay': 0.0001,
        'lr': 0.00002, 'start_lr': 0.0, 'final_lr': 0.0, 'warmup': 1,
    }],
})
cfg['model_kwargs']['checkpoint'] = '${CHECKPOINT}'
cfg['model_kwargs']['module_name'] = 'evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar'
cfg['experiment']['classifier'] = {'num_probe_blocks': 4, 'num_heads': 16}
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
        'warm_start_at_init': (not resume),
        'load_checkpoint_path': '${ENCODER_LORA_CKPT}',
        'checkpoint_path': '${RUN_DIR}/encoder_lora_latest.pt',
    },
    # Keep joint-v2 predictor adapters in the forward pass, but do not train them.
    'predictor_lora': {
        'enabled': True, 'rank': 8, 'alpha': 16.0, 'dropout': 0.05,
        'last_n_blocks': 0, 'lr_mult': 0.5, 'weight_decay': 0.0001,
        'activation_checkpointing': True,
        'target_suffixes': ['attn.qkv', 'attn.proj'],
        'warm_start_at_init': True,
        'freeze': True,
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
        'fusion': {
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
            'imu_hidden_dim': 128,
            # 0.25 * 2e-5 ≈ 5e-6 (same absolute fusion LR as earlier 0.05 * 1e-4).
            'lr_mult': 0.25,
            'weight_decay': 0.0001,
            'dropout': 0.0,
            **({'load_checkpoint_path': '${RUN_DIR}/tri_modal_fusion_latest.pt'} if resume else {}),
        },
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
print(f'Config written: {config_path} (resume={resume})')
PY

cd "${VJEPA_ROOT}"
"${PYTHON}" -m evals.main --fname "${CONFIG_PATH}" --devices cuda:0

echo "Tri-modal Stage-2 (from joint v2) complete: $(date)"
echo "Checkpoints: ${RUN_DIR}/{best.pt,encoder_lora_best.pt,tri_modal_fusion_best.pt,predictor_lora_best.pt}"
