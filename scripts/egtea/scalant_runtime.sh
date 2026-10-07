#!/bin/bash
# Runtime/path plumbing shared by the copied experiment launchers.
# Model, annotation and scoring settings remain in the Python adapter.

export SCALANT_WORKSPACE_ROOT=${SCALANT_WORKSPACE_ROOT:-${PROJECT_ROOT}}

scalant_exec() {
  local device_mode=${1:?expected cpu or gpu}
  shift
  mkdir -p "${PROJECT_ROOT}/outputs/b19_scalant/logs"
  case "${SCALANT_RUNTIME:-singularity}" in
    native) "$@" ;;
    singularity)
      local options=(--no-home -B "${SCALANT_BINDS:-/scratch,/share}" --overlay "${OVERLAY}:ro")
      if [ "$device_mode" = gpu ]; then options=(--nv "${options[@]}"); fi
      singularity exec "${options[@]}" "$IMAGE" "$@"
      ;;
    *) printf 'Unknown SCALANT_RUNTIME: %s\n' "$SCALANT_RUNTIME" >&2; return 2 ;;
  esac
}

scalant_activate() {
  export PYTHONNOUSERSITE=1
  if [ "${SCALANT_RUNTIME:-singularity}" = singularity ]; then
    source "${SCALANT_BOOTSTRAP:-/ext3/env.sh}"
    conda activate "${SCALANT_CONDA_ENV:-/ext3/envs/EgoLifeExp}"
  elif [ -n "${SCALANT_ENV_SETUP:-}" ]; then
    source "$SCALANT_ENV_SETUP"
  fi
}
