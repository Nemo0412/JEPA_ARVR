#!/usr/bin/env bash
# Val-only 1s matrix: native scope + single_head (auto-pick head by native action Top-3).
#
#   metric_scope=native
#   metric_aggregation=single_head  — one head for verb/noun/action, chosen on native logits
#
# Usage:
#   bash scripts/submit_native_singlehead_4way_matrix.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SLURM_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_lora_valonly_dump.slurm"

submit_native() {
  local tag="$1"
  local source_yaml="$2"
  local exports="ALL,SOURCE_YAML=${source_yaml},VAL_TAG=${tag}"
  exports+=",ANTICIPATION_SEC=1.0,LORA_VAL_METRIC_SCOPE=native"
  exports+=",LORA_VAL_METRIC_AGGREGATION=single_head"
  exports+=",VALDUMP_PREDICTION_DUMP=0,EVAL_BATCH_SIZE=1,EVAL_VAL_NUM_WORKERS=2"
  echo "[submit] ${tag} scope=native aggregation=single_head (auto head pick)"
  sbatch --export="${exports}" "${SLURM_SCRIPT}"
}

submit_native valdump-b1-native-singlehead-1s \
  configs/generated/valonly_dump/valdump-b1-clean-native-standard-1s.yaml

submit_native valdump-b5-binary-native-singlehead-1s \
  configs/generated/valonly_dump/valdump-b5-binary-map-native-standard-1s.yaml

submit_native valdump-b10-gaze-pose-gru-native-singlehead-1s \
  configs/generated/hdepic_lora_binary_gaze_pose_rnn_train.yaml

submit_native valdump-b11-matrix-native-singlehead-1s \
  configs/generated/valonly_dump/valdump-b11-matrix-normal-filtered-1s.yaml
