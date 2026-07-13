#!/bin/bash
# Monitor P01 stage-1 (video / gaze+pose) and auto-submit predictor stage-2 on completion.
# Resubmits stage-1 on low-GPU-util cancel (resume_checkpoint=True in submit scripts).

set -uo pipefail

INTERVAL=900
LOG="/scratch/ll5914/logs/monitor_p01.log"
STATE="/scratch/ll5914/logs/monitor_p01_state.env"
PIPELINE_VIDEO="/home/ll5914/Jepa_yifan/JEPA_ARVR/scripts/submit_p01_video_enc_joint_ll5914.slurm"
PIPELINE_GAZE="/home/ll5914/Jepa_yifan/JEPA_ARVR/scripts/submit_p01_gazepose_joint_ll5914.slurm"
PIPELINE_VIDEO_PRED="/home/ll5914/Jepa_yifan/JEPA_ARVR/scripts/submit_p01_video_predictor_ll5914.slurm"
PIPELINE_GAZE_PRED="/home/ll5914/Jepa_yifan/JEPA_ARVR/scripts/submit_p01_gazepose_predictor_ll5914.slurm"
STAGE1_VIDEO_DIR="/scratch/ll5914/experiments/p01_video_enc_clip/action_anticipation_frozen/p01-video-enc-clip-vitl16-256-10ep"
STAGE1_GAZE_DIR="/scratch/ll5914/experiments/p01_gazepose_clip/action_anticipation_frozen/p01-gazepose-clip-vitl16-256-10ep"
STAGE1_VIDEO_DONE="${STAGE1_VIDEO_DIR}/.stage1_complete"
STAGE1_GAZE_DONE="${STAGE1_GAZE_DIR}/.stage1_complete"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

save_state() {
  {
    echo "JID_VIDEO=${JID_VIDEO:-}"
    echo "JID_GAZE=${JID_GAZE:-}"
    echo "JID_VIDEO_PRED=${JID_VIDEO_PRED:-}"
    echo "JID_GAZE_PRED=${JID_GAZE_PRED:-}"
    echo "VIDEO_PRED_SUBMITTED=${VIDEO_PRED_SUBMITTED:-0}"
    echo "GAZE_PRED_SUBMITTED=${GAZE_PRED_SUBMITTED:-0}"
    echo "VIDEO_STAGE1_DONE=${VIDEO_STAGE1_DONE:-0}"
    echo "GAZE_STAGE1_DONE=${GAZE_STAGE1_DONE:-0}"
  } > "$STATE"
}

load_state() {
  JID_VIDEO=""; JID_GAZE=""; JID_VIDEO_PRED=""; JID_GAZE_PRED=""
  VIDEO_PRED_SUBMITTED=0; GAZE_PRED_SUBMITTED=0
  VIDEO_STAGE1_DONE=0; GAZE_STAGE1_DONE=0
  source "$STATE" 2>/dev/null || true
}

job_state() { sacct -j "$1" -n -X -o State 2>/dev/null | head -1 | awk '{print $1}'; }
job_queue() { squeue -j "$1" -h -o "%T" 2>/dev/null; }
job_reason() { squeue -j "$1" -h -o "%R" 2>/dev/null || true; }
job_elapsed() { squeue -j "$1" -h -o "%M" 2>/dev/null || sacct -j "$1" -n -X -o Elapsed 2>/dev/null | head -1; }

sync_jids_from_queue() {
  local v g vp gp
  v=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk '$2=="p01_video_enc"{print $1; exit}')
  g=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk '$2=="p01_gazepose"{print $1; exit}')
  vp=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk '$2=="p01_video_pred"{print $1; exit}')
  gp=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk '$2=="p01_gazepose_pred"{print $1; exit}')
  [[ -n "$v" && "${VIDEO_STAGE1_DONE:-0}" != "1" ]] && JID_VIDEO="$v"
  [[ -n "$g" && "${GAZE_STAGE1_DONE:-0}" != "1" ]] && JID_GAZE="$g"
  [[ -n "$vp" ]] && JID_VIDEO_PRED="$vp"
  [[ -n "$gp" ]] && JID_GAZE_PRED="$gp"
}

mark_stage1_done() {
  local label="$1"
  if [[ "$label" == "gaze" ]]; then
    GAZE_STAGE1_DONE=1
    touch "$STAGE1_GAZE_DONE" 2>/dev/null || true
  else
    VIDEO_STAGE1_DONE=1
    touch "$STAGE1_VIDEO_DONE" 2>/dev/null || true
  fi
}

stage1_ready() {
  local label="$1" dir probe enc adapter
  if [[ "$label" == "gaze" ]]; then
    dir="$STAGE1_GAZE_DIR"
    probe="${dir}/best.pt"; enc="${dir}/encoder_lora_best.pt"
    adapter="${dir}/binary_input_adapter_best.pt"
    [[ -f "$probe" && -f "$enc" && -f "$adapter" ]]
  else
    dir="$STAGE1_VIDEO_DIR"
    probe="${dir}/best.pt"; enc="${dir}/encoder_lora_best.pt"
    [[ -f "$probe" && -f "$enc" ]]
  fi
}

refresh_stage1_done_flags() {
  [[ "${VIDEO_STAGE1_DONE:-0}" == "1" || -f "$STAGE1_VIDEO_DONE" ]] && VIDEO_STAGE1_DONE=1
  [[ "${GAZE_STAGE1_DONE:-0}" == "1" || -f "$STAGE1_GAZE_DONE" ]] && GAZE_STAGE1_DONE=1
  if [[ "${VIDEO_STAGE1_DONE:-0}" != "1" ]] && stage1_ready "video" && [[ -f "$STAGE1_VIDEO_DONE" ]]; then
    VIDEO_STAGE1_DONE=1
  fi
  if [[ "${GAZE_STAGE1_DONE:-0}" != "1" ]] && stage1_ready "gaze" && [[ -f "$STAGE1_GAZE_DONE" ]]; then
    GAZE_STAGE1_DONE=1
  fi
}

latest_err_for_job() {
  local jid="$1" jname="$2"
  ls -t /scratch/ll5914/logs/${jname}_${jid}.err 2>/dev/null | head -1
}

classify_cancel_reason() {
  local jid="$1" state="$2" jname="$3"
  [[ "$(echo "${state}" | tr '[:lower:]' '[:upper:]')" == CANCELLED* ]] || return 0
  local err_file
  err_file="$(latest_err_for_job "$jid" "$jname")"
  if [[ -f "${err_file}" ]] && grep -q 'DUE to SIGNAL Terminated' "${err_file}" \
      && ! grep -qiE 'Traceback|CUDA out of memory|OutOfMemoryError|Killed' "${err_file}"; then
    echo "low_gpu_util"; return 0
  fi
  echo "cancel_other"
}

submit_predictor_stage2() {
  local label="$1" pipeline="$2" stage1_dir="$3"
  local flag_var jid_var
  if [[ "$label" == "gaze" ]]; then
    flag_var=GAZE_PRED_SUBMITTED; jid_var=JID_GAZE_PRED
  else
    flag_var=VIDEO_PRED_SUBMITTED; jid_var=JID_VIDEO_PRED
  fi
  [[ "${!flag_var}" == "1" ]] && return 0
  if ! stage1_ready "$label"; then
    log "[${label}-pred] defer: stage-1 best checkpoints missing in ${stage1_dir}"
    return 0
  fi
  log "[${label}-pred] SUBMIT predictor stage-2 from ${stage1_dir}"
  local NEWJID
  NEWJID=$(STAGE1_DIR="${stage1_dir}" sbatch "$pipeline" | awk '{print $NF}')
  eval "${jid_var}=${NEWJID}"
  eval "${flag_var}=1"
  save_state
  log "[${label}-pred] SUBMIT OK JID=${NEWJID}"
}

resubmit_stage1() {
  local label="$1" pipeline="$2"
  if [[ "$label" == "video" && "${VIDEO_STAGE1_DONE:-0}" == "1" ]]; then return 0; fi
  if [[ "$label" == "gaze" && "${GAZE_STAGE1_DONE:-0}" == "1" ]]; then return 0; fi
  log "[${label}] RESUBMIT stage-1 (resume)"
  local NEWJID
  NEWJID=$(sbatch "$pipeline" | awk '{print $NF}')
  if [[ "$label" == "video" ]]; then JID_VIDEO=$NEWJID; else JID_GAZE=$NEWJID; fi
  save_state
  log "[${label}] RESUBMIT OK JID=${NEWJID}"
}

stage1_training_done() {
  local label="$1" jid="$2" jname="$3" st="$4"
  stage1_ready "$label" || return 1
  [[ "$st" == "COMPLETED" ]] && return 0
  [[ "$st" != "FAILED" ]] && return 1
  local out="/scratch/ll5914/logs/${jname}_${jid}.out"
  local err
  err="$(latest_err_for_job "$jid" "$jname")"
  [[ -f "$out" ]] && grep -qE 'Early stopping|training complete:' "$out" 2>/dev/null && return 0
  [[ -f "$err" ]] && grep -q 'outside the classifier action map' "$err" 2>/dev/null && return 0
  return 1
}

check_stage1_job() {
  local label="$1" jid="$2" jname="$3" pipeline="$4"
  [[ -z "$jid" ]] && return 0
  if [[ "$label" == "video" && "${VIDEO_STAGE1_DONE:-0}" == "1" ]]; then return 0; fi
  if [[ "$label" == "gaze" && "${GAZE_STAGE1_DONE:-0}" == "1" ]]; then return 0; fi

  local Q ST EL REASON
  Q=$(job_queue "$jid")
  ST=$(job_state "$jid")
  EL=$(job_elapsed "$jid")
  REASON=$(job_reason "$jid")
  log "CHECK [${label}]: jid=${jid} queue=${Q:-none} sacct=${ST:-unknown} elapsed=${EL:-0} reason=${REASON:-n/a}"

  if [[ "$Q" == "RUNNING" || "$Q" == "PENDING" ]]; then return 0; fi

  if stage1_training_done "$label" "$jid" "$jname" "$ST"; then
    log "[${label}] stage-1 finished (state=${ST}) jid=${jid}"
    mark_stage1_done "$label"
    if [[ "$label" == "video" ]]; then
      submit_predictor_stage2 "video" "$PIPELINE_VIDEO_PRED" "$STAGE1_VIDEO_DIR"
    else
      submit_predictor_stage2 "gaze" "$PIPELINE_GAZE_PRED" "$STAGE1_GAZE_DIR"
    fi
    return 0
  fi

  if [[ -n "$ST" && "$ST" != "RUNNING" && "$ST" != "PENDING" ]]; then
    local cancel_kind
    cancel_kind=$(classify_cancel_reason "$jid" "$ST" "$jname")
    log "[${label}] terminal state=${ST} (${cancel_kind}) — resubmit stage-1"
    resubmit_stage1 "$label" "$pipeline"
  fi
}

check_pred_job() {
  local label="$1" jid="$2" jname="$3"
  [[ -z "$jid" ]] && return 0
  local Q ST
  Q=$(job_queue "$jid")
  ST=$(job_state "$jid")
  log "CHECK [${label}-pred]: jid=${jid} queue=${Q:-none} sacct=${ST:-unknown}"
}

mkdir -p /scratch/ll5914/logs
LOCK="/scratch/ll5914/logs/monitor_p01.lock"
if [[ -f "$LOCK" ]]; then
  oldpid=$(cat "$LOCK" 2>/dev/null || true)
  if [[ -n "$oldpid" ]] && kill -0 "$oldpid" 2>/dev/null; then
    log "Another P01 monitor already running (pid=$oldpid). Exiting."
    exit 0
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

load_state
refresh_stage1_done_flags
JID_VIDEO=${JID_VIDEO:-13251652}
JID_GAZE=${JID_GAZE:-13251653}
sync_jids_from_queue
save_state
log "P01 monitor started video=${JID_VIDEO} gaze=${JID_GAZE} video_pred=${JID_VIDEO_PRED:-} gaze_pred=${JID_GAZE_PRED:-} pid=$$"

while true; do
  load_state
  refresh_stage1_done_flags
  sync_jids_from_queue

  check_stage1_job "video" "$JID_VIDEO" "p01_video_enc" "$PIPELINE_VIDEO"
  check_stage1_job "gaze" "$JID_GAZE" "p01_gazepose" "$PIPELINE_GAZE"
  check_pred_job "video" "$JID_VIDEO_PRED" "p01_video_pred"
  check_pred_job "gaze" "$JID_GAZE_PRED" "p01_gazepose_pred"

  if [[ "${VIDEO_STAGE1_DONE:-0}" == "1" && "${VIDEO_PRED_SUBMITTED:-0}" != "1" ]]; then
    submit_predictor_stage2 "video" "$PIPELINE_VIDEO_PRED" "$STAGE1_VIDEO_DIR"
  fi
  if [[ "${GAZE_STAGE1_DONE:-0}" == "1" && "${GAZE_PRED_SUBMITTED:-0}" != "1" ]]; then
    submit_predictor_stage2 "gaze" "$PIPELINE_GAZE_PRED" "$STAGE1_GAZE_DIR"
  fi

  save_state

  V_DONE=${VIDEO_STAGE1_DONE:-0}
  G_DONE=${GAZE_STAGE1_DONE:-0}
  VP_ST=$(job_state "${JID_VIDEO_PRED:-}")
  GP_ST=$(job_state "${JID_GAZE_PRED:-}")
  if [[ "$V_DONE" == "1" && "$G_DONE" == "1" ]]; then
    if [[ -z "${JID_VIDEO_PRED:-}" || "$VP_ST" == "COMPLETED" ]]; then
      if [[ -z "${JID_GAZE_PRED:-}" || "$GP_ST" == "COMPLETED" ]]; then
        log "P01 stage-1 and stage-2 all done. Monitor exiting."
        exit 0
      fi
    fi
  fi

  sleep "$INTERVAL"
done
