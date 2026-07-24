#!/bin/bash
#SBATCH --job-name=p01_video_mtp
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=384GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ll5914/logs/p01_video_mtp_%j.out
#SBATCH --error=/scratch/ll5914/logs/p01_video_mtp_%j.err

set -euo pipefail

# P01 video-only Multi-Time Prediction (MTP):
#   predict action @ 1s / 5s / 10s with short→long cascade.
# Warm-start from best video joint-v2 heads ckpt (val-action-top5=40.44% @ep3).
# Single-horizon path untouched when mtp.enabled=false.

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"

ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split"
VIDEO_ROOT="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
HDEPIC_ANN_ROOT="/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"

# Best video-only joint (heads) checkpoint.
VIDEO_BEST_DIR="${VIDEO_BEST_DIR:-/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep}"
VIDEO_PROBE="${VIDEO_BEST_DIR}/best.pt"
VIDEO_ENCODER_LORA="${VIDEO_BEST_DIR}/encoder_lora_best.pt"
VIDEO_PREDICTOR_LORA="${VIDEO_BEST_DIR}/predictor_lora_best.pt"

OUTPUT_DIR="/scratch/ll5914/experiments/p01_video_mtp_1_5_10"
CONFIG_PATH="/scratch/ll5914/logs/p01_video_mtp_1_5_10_config.yaml"
TAG="p01-video-mtp-1-5-10-vitl16-256-8ep"

for f in "${VIDEO_PROBE}" "${VIDEO_ENCODER_LORA}" "${VIDEO_PREDICTOR_LORA}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: video best checkpoint missing: ${f}" >&2
        exit 1
    fi
done

export MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))
echo "========================================================"
echo "P01 video MTP (1s→5s→10s) from video joint-v2 best"
echo "Job=${SLURM_JOB_ID:-manual} MASTER_PORT=${MASTER_PORT}"
echo "Video best: ${VIDEO_BEST_DIR}"
echo "Out: ${OUTPUT_DIR}"
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
cfg['tasks_per_node'] = 1
cfg['cpus_per_task'] = 24
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
    'num_workers': 10, 'val_num_workers': 6, 'clip_balanced': True,
    'resolution': 256, 'frames_per_clip': 32, 'frames_per_second': 8,
    # Clip construction stays on the 1s anticipation protocol.
    'anticipation_time_sec': [1.0, 1.0],
    'train_anticipation_time_sec': [1.0, 1.0],
})
opt = cfg['experiment']['optimization']
# Heads LR matches joint-v2 recipe; predictor LoRA lr_mult=0.5.
# bs=8: 5s/10s use sliding-window AR (extra predictor forwards vs 1s-only).
opt.update({'batch_size': 8, 'num_epochs': 8, 'use_bfloat16': True,
    'use_focal_loss': False, 'val_every_epochs': 1, 'best_metric': 'val-action-top5',
    'early_stopping_patience': 3, 'grad_clip': 1.0,
    'multihead_kwargs': [{'weight_decay': 0.0001, 'final_weight_decay': 0.0001,
        'lr': 0.00002, 'start_lr': 0.0, 'final_lr': 0.0, 'warmup': 1}]})
cfg['model_kwargs']['checkpoint'] = '${CHECKPOINT}'
cfg['experiment']['lora'] = {
    'enabled': True, 'rank': 8, 'alpha': 16.0, 'dropout': 0.05,
    'probe_train_mode': 'full',
    'train_heads': True,
    'freeze_pooler': True,
    'load_probe_heads': True,
    'pretrained_probe': '${VIDEO_PROBE}',
    'align_reference_metrics': True, 'val_metric_scope': 'native',
    'val_metric_aggregation': 'metric_wise_max', 'class_space': 'train_only',
    'temporal_sampling': 'phd_reference', 'output_regularization': {'enabled': False},
    'hdepic_reference': {
        'annotations_pkl': '${HDEPIC_ANN_ROOT}/HD_EPIC_Narrations.pkl',
        'verb_classes_csv': '${HDEPIC_ANN_ROOT}/HD_EPIC_verb_classes.csv',
        'noun_classes_csv': '${HDEPIC_ANN_ROOT}/HD_EPIC_noun_classes.csv',
    },
    'encoder_lora': {
        'enabled': True, 'rank': 8, 'alpha': 16.0, 'dropout': 0.05,
        'last_n_blocks': 0, 'lr_mult': 0.5, 'weight_decay': 0.0001,
        'activation_checkpointing': True, 'target_suffixes': ['attn.qkv', 'attn.proj'],
        'load_checkpoint_path': '${VIDEO_ENCODER_LORA}',
        'warm_start_at_init': True, 'freeze': True,
    },
    'predictor_lora': {
        'enabled': True, 'rank': 8, 'alpha': 16.0, 'dropout': 0.05,
        'last_n_blocks': 0, 'lr_mult': 0.5, 'weight_decay': 0.0001,
        'activation_checkpointing': True, 'target_suffixes': ['attn.qkv', 'attn.proj'],
        'load_checkpoint_path': '${VIDEO_PREDICTOR_LORA}',
        'warm_start_at_init': True, 'freeze': False,
    },
    'gaze': {'mode': 'none'},
    # Legacy Multi-Time Prediction: predictor multi-horizon + cascaded heads (1→5→10).
    'mtp': {
        'enabled': True,
        'head_type': 'cascaded',
        'backbone_mode': 'multi_predict',
        'label_mode': 'primary_from_sample',
        'horizons_sec': [1.0, 5.0, 10.0],
        'loss_weights': [1.0, 0.7, 0.5],
        'primary_horizon_sec': 1.0,
        'condition_mode': 'feature_token',
        'teacher_forcing': True,
    },
}
config_path.parent.mkdir(parents=True, exist_ok=True)
Path('${OUTPUT_DIR}').mkdir(parents=True, exist_ok=True)
with config_path.open('w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f'Config: {config_path}')
print('MTP horizons=[1,5,10] loss_weights=[1.0,0.7,0.5] primary=1s')
print(f'Warm-start probe/enc/pred from {Path("${VIDEO_BEST_DIR}").name}')
PY

cd "${VJEPA_ROOT}"
"${PYTHON}" -m evals.main --fname "${CONFIG_PATH}" --devices cuda:0

echo "P01 video MTP complete: $(date)"
