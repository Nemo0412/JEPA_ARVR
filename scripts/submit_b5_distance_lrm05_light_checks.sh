#!/bin/bash
set -euo pipefail

# Submit the removable light-check matrix for the non-identity B5
# distance-transform + adapter-LR0.5 checkpoint from training job 9801351.
#
# Run this from the HPC login node:
#   cd /path/to/VJEPA2-EXP
#   bash scripts/submit_b5_distance_lrm05_light_checks.sh

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
DIST_TAG="${DIST_TAG:-hdepic-lora-binary-input-adapter-distance-lrm05-r64-h8-bs2-w12}"
OLD_TAG="${OLD_TAG:-hdepic-lora-binary-input-adapter-r64-h8-dlmap-multihead-bs2-w12-gazefixed-gradisolate}"
DIST_YAML="${DIST_YAML:-${PROJECT_ROOT}/configs/generated/hdepic_lora_binary_input_adapter_distance_lrmult05.yaml}"
OLD_YAML="${OLD_YAML:-${PROJECT_ROOT}/configs/generated/future_latent_compare/path_y_b5_binary_adapter_10s.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"

STANDARD_MODULE="evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar"
ROLLOUT_MODULE="app.hdepic_lora_action_anticipation.modelcustom.vit_encoder_predictor_rollout"
FM_WRAPPER="${PROJECT_ROOT}/scripts/run_hdepic_future_latent_failure_modes.slurm"
VAL_WRAPPER="${PROJECT_ROOT}/scripts/run_hdepic_lora_valonly_dump.slurm"
RESCORE_WRAPPER="${PROJECT_ROOT}/scripts/run_hdepic_rescore_window_cpu.slurm"

for path in "${DIST_YAML}" "${OLD_YAML}" "${FM_WRAPPER}" "${VAL_WRAPPER}" "${RESCORE_WRAPPER}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 2
    fi
done

fm_1s_job=$(sbatch --parsable \
    --export=ALL,CONFIG_PATH="${DIST_YAML}",OUT_DIR="${OUTPUT_ROOT}/future_latent_failure_modes_native_1s/B5-binary-distance-lrm05-1s",HORIZON=1.0,METRIC_SCOPE=native \
    "${FM_WRAPPER}")
fm_10s_job=$(sbatch --parsable \
    --export=ALL,CONFIG_PATH="${DIST_YAML}",OUT_DIR="${OUTPUT_ROOT}/future_latent_failure_modes_native/B5-binary-distance-lrm05-10s",HORIZON=10.0,METRIC_SCOPE=native \
    "${FM_WRAPPER}")

old_standard_1s_job=$(sbatch --parsable \
    --export=ALL,SOURCE_YAML="${OLD_YAML}",VAL_TAG=valdump-b5-old-native-standard-1s,ANTICIPATION_SEC=1.0,LORA_VAL_METRIC_SCOPE=native,MODEL_MODULE_NAME="${STANDARD_MODULE}",DUMP_PATH="${OUTPUT_ROOT}/action_anticipation_frozen/${OLD_TAG}/val_predictions_native_standard_1s.csv" \
    "${VAL_WRAPPER}")
dist_standard_1s_job=$(sbatch --parsable \
    --export=ALL,SOURCE_YAML="${DIST_YAML}",VAL_TAG=valdump-b5-distance-lrm05-native-standard-1s,ANTICIPATION_SEC=1.0,LORA_VAL_METRIC_SCOPE=native,MODEL_MODULE_NAME="${STANDARD_MODULE}",DUMP_PATH="${OUTPUT_ROOT}/action_anticipation_frozen/${DIST_TAG}/val_predictions_native_standard_1s.csv" \
    "${VAL_WRAPPER}")

old_window_10s_job=$(sbatch --parsable \
    --export=ALL,SOURCE_YAML="${OLD_YAML}",VAL_TAG=valdump-b5-old-native-ar10s-adapterloaded,ANTICIPATION_SEC=10.0,LORA_VAL_METRIC_SCOPE=native,MODEL_MODULE_NAME="${ROLLOUT_MODULE}",MODEL_MAX_ROLLOUT_STEPS=512,DUMP_PATH="${OUTPUT_ROOT}/action_anticipation_frozen/${OLD_TAG}/val_predictions_native_ar10s_adapterloaded.csv" \
    "${VAL_WRAPPER}")
dist_window_10s_job=$(sbatch --parsable \
    --export=ALL,SOURCE_YAML="${DIST_YAML}",VAL_TAG=valdump-b5-distance-lrm05-native-ar10s,ANTICIPATION_SEC=10.0,LORA_VAL_METRIC_SCOPE=native,MODEL_MODULE_NAME="${ROLLOUT_MODULE}",MODEL_MAX_ROLLOUT_STEPS=512,DUMP_PATH="${OUTPUT_ROOT}/action_anticipation_frozen/${DIST_TAG}/val_predictions_native_ar10s.csv" \
    "${VAL_WRAPPER}")

rescore_job=$(sbatch --parsable \
    --dependency="afterok:${old_window_10s_job}:${dist_window_10s_job}" \
    --export=ALL,PREDICTION_1="${OUTPUT_ROOT}/action_anticipation_frozen/${OLD_TAG}/val_predictions_native_ar10s_adapterloaded.csv",LABEL_1=B5-old-binary-native-ar10s-adapterloaded,PREDICTION_2="${OUTPUT_ROOT}/action_anticipation_frozen/${DIST_TAG}/val_predictions_native_ar10s.csv",LABEL_2=B5-distance-lrm05-native-ar10s,WINDOW_SEC=5.0,OUT_DIR="${PROJECT_ROOT}/outputs/rescore_window_5s_native_vjepa2_b5_distance_lrm05" \
    "${RESCORE_WRAPPER}")

cat <<EOF
Submitted B5 distance-lrm05 light-check matrix:
  future latent 1s:          ${fm_1s_job}
  future latent 10s:         ${fm_10s_job}
  old B5 standard native 1s: ${old_standard_1s_job}
  new B5 standard native 1s: ${dist_standard_1s_job}
  old B5 native AR 10s dump: ${old_window_10s_job}
  new B5 native AR 10s dump: ${dist_window_10s_job}
  dependent 5s rescore:      ${rescore_job}
EOF
