#!/bin/bash
#SBATCH --job-name=tri_modal_fusion
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=384GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ll5914/logs/tri_modal_fusion_%j.out
#SBATCH --error=/scratch/ll5914/logs/tri_modal_fusion_%j.err

set -euo pipefail

# Tri-modal projected cross-attention fusion @ 1s anticipation.
# Trainable: probe (full) + encoder LoRA + gaze/IMU encoders + fusion cross-attn.
# Frozen: base ViT encoder weights, predictor.

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"

ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations"
VIDEO_ROOT="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
HDEPIC_ANN_ROOT="/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"
OUTPUT_DIR="/scratch/ll5914/experiments/tri_modal_fusion_legacy_1s"
CONFIG_PATH="/scratch/ll5914/logs/tri_modal_fusion_legacy_1s_config.yaml"
TAG="tri-modal-fusion-legacy-vitl16-256-10ep-2xh100"

GAZE_EXTRACT_ROOT="/scratch/ll5914/datasets/HD-EPIC/_gaze_extract"
GAZE_ROOT="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze"
GAZE_SYNC_ROOT="/scratch/ll5914/datasets/HD-EPIC/Videos"
POSE_SLAM_ROOT="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze/P01/SLAM/multi"
POSE_MAPPING_JSON="${POSE_SLAM_ROOT}/vrs_to_multi_slam.json"

mkdir -p /scratch/ll5914/logs "${OUTPUT_DIR}"

echo "========================================================"
echo "Tri-modal fusion: probe full + encoder LoRA + gaze/IMU/fusion (2x H100)"
echo "Job  : ${SLURM_JOB_ID:-manual}  Node: $(hostname)"
echo "Start: $(date)"
echo "========================================================"

export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

"${PYTHON}" - <<PY
import sys, yaml
from pathlib import Path
sys.path.insert(0, '${PROJECT_ROOT}')
sys.path.insert(0, '${VJEPA_ROOT}')
template_path = Path('${VJEPA_ROOT}/configs/eval/vitl/ek100.yaml')
config_path   = Path('${CONFIG_PATH}')
with template_path.open() as f:
    cfg = yaml.safe_load(f)
cfg['nodes'] = 1
cfg['tasks_per_node'] = 2
cfg['cpus_per_task'] = 16
cfg['tag']  = '${TAG}'
cfg['eval_name'] = 'app.hdepic_lora_action_anticipation'
cfg['folder'] = '${OUTPUT_DIR}'
cfg['resume_checkpoint'] = False
cfg['val_only'] = False
data = cfg['experiment']['data']
data.update({
    'dataset': 'EK100', 'file_format': 1,
    'base_path': '${VIDEO_ROOT}',
    'dataset_train': '${ANN_DIR}/HD_EPIC_train_vjepa.csv',
    'dataset_val': '${ANN_DIR}/HD_EPIC_val_vjepa.csv',
    'num_workers': 4, 'val_num_workers': 2, 'clip_balanced': True,
    'resolution': 256, 'frames_per_clip': 32, 'frames_per_second': 8,
    'anticipation_time_sec': [1.0, 1.0],
    'train_anticipation_time_sec': [0.25, 1.75],
})
opt = cfg['experiment']['optimization']
opt.update({
    'batch_size': 6, 'num_epochs': 10, 'use_bfloat16': True,
    'use_focal_loss': False, 'val_every_epochs': 1, 'best_metric': 'val-action-top5',
    'early_stopping_patience': 3, 'grad_clip': 1.0,
    'multihead_kwargs': [{
        'weight_decay': 0.0001, 'final_weight_decay': 0.0001,
        'lr': 0.0001, 'start_lr': 0.0, 'final_lr': 0.0, 'warmup': 2,
    }],
})
cfg['model_kwargs']['checkpoint'] = '${CHECKPOINT}'
cfg['model_kwargs']['module_name'] = 'app.hdepic_lora_action_anticipation.modelcustom.vit_encoder_predictor_rollout'
cfg['model_kwargs']['wrapper_kwargs'] = {'num_steps': 1, 'return_mode': 'observed_plus_target'}
cfg['experiment']['lora'] = {
    'enabled': True, 'rank': 8, 'alpha': 16.0, 'dropout': 0.05,
    'probe_train_mode': 'full', 'train_heads': True, 'pretrained_probe': '',
    'align_reference_metrics': True, 'val_metric_scope': 'native',
    'val_metric_aggregation': 'metric_wise_max',
    'class_space': 'train_only', 'temporal_sampling': 'legacy',
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
    },
    'predictor_lora': {'enabled': False},
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
            'lr_mult': 0.05,
            'weight_decay': 0.0001,
            'dropout': 0.0,
        },
        'pose': {
            'enabled': True,
            'slam_root': '${POSE_SLAM_ROOT}',
            'mapping_json': '${POSE_MAPPING_JSON}',
            'feature_set': 'pose_6d',
            'quality_min': 0.0,
            'history_sec': 0,
            'interframe_k_max': 128,
        },
    },
}
config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open('w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f'Config written: {config_path}')
PY

cd "${VJEPA_ROOT}"
"${PYTHON}" -m evals.main --fname "${CONFIG_PATH}" --devices cuda:0 cuda:1

RUN_DIR="${OUTPUT_DIR}/action_anticipation_frozen/${TAG}"
echo "Tri-modal fusion training complete: $(date)"
echo "Checkpoints: ${RUN_DIR}/{best.pt,encoder_lora_best.pt,tri_modal_fusion_best.pt}"
