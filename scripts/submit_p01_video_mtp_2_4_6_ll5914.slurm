#!/bin/bash
#SBATCH --job-name=p01_mtp_246
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=384GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ll5914/logs/p01_mtp_246_%j.out
#SBATCH --error=/scratch/ll5914/logs/p01_mtp_246_%j.err

set -euo pipefail

# P01 video-only MTP (communicating MLP @ 2s/4s/6s):
#   one shared anticipative forward → 3 private MLPs that mutually communicate
#   (self-attn over horizons) → shared verb/noun/action heads.
# No RNN / no multi-horizon AR predictor rollout.
# Warm-start from video joint-v2 heads (val-action-top5=40.44% @ep3).
# Opt-in only: mtp.enabled=false leaves the single-horizon path untouched.
#
# Set VAL_ONLY=1 to run validation-only against RUN_DIR best/latest.

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"

ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split"
VIDEO_ROOT="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
HDEPIC_ANN_ROOT="/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments"
CHECKPOINT="/scratch/ll5914/models/vjepa2/vitl.pt"

VIDEO_BEST_DIR="${VIDEO_BEST_DIR:-/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep}"
VIDEO_PROBE="${VIDEO_BEST_DIR}/best.pt"
VIDEO_ENCODER_LORA="${VIDEO_BEST_DIR}/encoder_lora_best.pt"
VIDEO_PREDICTOR_LORA="${VIDEO_BEST_DIR}/predictor_lora_best.pt"

OUTPUT_DIR="/scratch/ll5914/experiments/p01_video_mtp_2_4_6"
TAG="p01-video-mtp-comm-mlp-2-4-6-vitl16-256-8ep"
RUN_DIR="${OUTPUT_DIR}/action_anticipation_frozen/${TAG}"
CONFIG_PATH="/scratch/ll5914/logs/p01_video_mtp_2_4_6_config.yaml"
VAL_ONLY="${VAL_ONLY:-0}"

for f in "${VIDEO_PROBE}" "${VIDEO_ENCODER_LORA}" "${VIDEO_PREDICTOR_LORA}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: video best checkpoint missing: ${f}" >&2
    exit 1
  fi
done

PROBE_CKPT="${VIDEO_PROBE}"
ENCODER_LORA_CKPT="${VIDEO_ENCODER_LORA}"
PREDICTOR_LORA_CKPT="${VIDEO_PREDICTOR_LORA}"
RESUME=0
if [[ "${VAL_ONLY}" == "1" ]]; then
  # Resume latest.pt so CommunicatingMLPMTPClassifier weights restore (not just base probe).
  if [[ -f "${RUN_DIR}/best.pt" ]]; then
    cp -f "${RUN_DIR}/best.pt" "${RUN_DIR}/latest.pt"
    for side in encoder_lora predictor_lora; do
      if [[ -f "${RUN_DIR}/${side}_best.pt" ]]; then
        cp -f "${RUN_DIR}/${side}_best.pt" "${RUN_DIR}/${side}_latest.pt"
      fi
    done
  fi
  if [[ ! -f "${RUN_DIR}/latest.pt" ]]; then
    echo "ERROR: VAL_ONLY=1 but no best/latest under ${RUN_DIR}" >&2
    exit 1
  fi
  PROBE_CKPT="${RUN_DIR}/latest.pt"
  ENCODER_LORA_CKPT="${RUN_DIR}/encoder_lora_latest.pt"
  PREDICTOR_LORA_CKPT="${RUN_DIR}/predictor_lora_latest.pt"
  RESUME=1
elif [[ -f "${RUN_DIR}/latest.pt" ]]; then
  PROBE_CKPT="${RUN_DIR}/latest.pt"
  ENCODER_LORA_CKPT="${RUN_DIR}/encoder_lora_latest.pt"
  PREDICTOR_LORA_CKPT="${RUN_DIR}/predictor_lora_latest.pt"
  RESUME=1
fi

export MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))
mkdir -p /scratch/ll5914/logs "${OUTPUT_DIR}" "${RUN_DIR}"

echo "========================================================"
echo "P01 video MTP communicating-MLP (2s/4s/6s)"
echo "Job=${SLURM_JOB_ID:-manual} VAL_ONLY=${VAL_ONLY} resume=${RESUME}"
echo "Probe: ${PROBE_CKPT}"
echo "Out: ${RUN_DIR}"
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
val_only = str('${VAL_ONLY}') == '1'
resume = bool(int('${RESUME}'))  # val_only also resumes latest to restore MTP MLPs
with template_path.open() as f:
    cfg = yaml.safe_load(f)
cfg['nodes'] = 1
cfg['tasks_per_node'] = 1
cfg['cpus_per_task'] = 24
cfg['tag']  = '${TAG}'
cfg['eval_name'] = 'app.hdepic_lora_action_anticipation'
cfg['folder'] = '${OUTPUT_DIR}'
cfg['resume_checkpoint'] = resume
cfg['val_only'] = val_only
data = cfg['experiment']['data']
data.update({
    'dataset': 'EK100', 'file_format': 1,
    'base_path': '${VIDEO_ROOT}',
    'dataset_train': '${ANN_DIR}/HD_EPIC_train_vjepa.csv',
    'dataset_val': '${ANN_DIR}/HD_EPIC_val_vjepa.csv',
    'num_workers': 10, 'val_num_workers': 6, 'clip_balanced': True,
    'resolution': 256, 'frames_per_clip': 32, 'frames_per_second': 8,
    # Clip construction stays on the standard 1s protocol; MTP labels look up 2/4/6s.
    'anticipation_time_sec': [1.0, 1.0],
    'train_anticipation_time_sec': [1.0, 1.0],
})
opt = cfg['experiment']['optimization']
opt.update({
    'batch_size': 8, 'num_epochs': 8, 'use_bfloat16': True,
    'use_focal_loss': False, 'val_every_epochs': 1, 'best_metric': 'val-action-top5',
    'early_stopping_patience': 3, 'grad_clip': 1.0,
    'multihead_kwargs': [{
        'weight_decay': 0.0001, 'final_weight_decay': 0.0001,
        'lr': 0.00002, 'start_lr': 0.0, 'final_lr': 0.0, 'warmup': 1,
    }],
})
cfg['model_kwargs']['checkpoint'] = '${CHECKPOINT}'
cfg['experiment']['lora'] = {
    'enabled': True, 'rank': 8, 'alpha': 16.0, 'dropout': 0.05,
    'probe_train_mode': 'full',
    'train_heads': True,
    'freeze_pooler': True,
    'load_probe_heads': True,
    'pretrained_probe': '${PROBE_CKPT}',
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
        'load_checkpoint_path': '${ENCODER_LORA_CKPT}',
        'warm_start_at_init': True, 'freeze': True,
        'checkpoint_path': '${RUN_DIR}/encoder_lora_latest.pt',
    },
    'predictor_lora': {
        'enabled': True, 'rank': 8, 'alpha': 16.0, 'dropout': 0.05,
        'last_n_blocks': 0, 'lr_mult': 0.5, 'weight_decay': 0.0001,
        'activation_checkpointing': True, 'target_suffixes': ['attn.qkv', 'attn.proj'],
        'load_checkpoint_path': '${PREDICTOR_LORA_CKPT}',
        'warm_start_at_init': True, 'freeze': False,
        'checkpoint_path': '${RUN_DIR}/predictor_lora_latest.pt',
    },
    'gaze': {'mode': 'none'},
    'mtp': {
        'enabled': True,
        'head_type': 'communicating_mlp',
        'backbone_mode': 'shared',
        'label_mode': 'lookup_all',
        'horizons_sec': [2.0, 4.0, 6.0],
        'loss_weights': [1.0, 0.7, 0.5],
        'primary_horizon_sec': 2.0,
        'comm_layers': 2,
        'comm_heads': 4,
    },
}
config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open('w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f'Config: {config_path} val_only={val_only} resume={resume}')
print('MTP communicating_mlp horizons=[2,4,6] primary=2s')
PY

cd "${VJEPA_ROOT}"
"${PYTHON}" -m evals.main --fname "${CONFIG_PATH}" --devices cuda:0

echo "P01 video MTP 2/4/6 done: $(date) VAL_ONLY=${VAL_ONLY}"
