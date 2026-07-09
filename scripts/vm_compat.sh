#!/usr/bin/env bash
# Shared hardware/library compatibility defaults for CROPI VM profiles.
#
# Source this from setup/install scripts, then call:
#   cropi_apply_vm_compat
# It only fills unset variables, so explicit user overrides still win.

cropi_detect_vm_profile() {
  if [[ -n "${CROPI_VM_PROFILE:-}" ]]; then
    echo "${CROPI_VM_PROFILE}"
    return
  fi

  local names=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '[:upper:]' '[:lower:]' || true)"
  fi

  if grep -qiE 'h100|h200|hopper|gh200' <<<"${names}"; then
    echo "h100"
  elif grep -qi 'a100' <<<"${names}"; then
    echo "a100"
  else
    echo "generic"
  fi
}

cropi_detect_driver_cuda() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -1
}

cropi_version_ge() {
  # usage: cropi_version_ge 12.4 12.2
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" == "$2" ]]
}

cropi_cuda_redist_default() {
  local profile="${1:-$(cropi_detect_vm_profile)}"
  local driver_cuda="${2:-$(cropi_detect_driver_cuda)}"

  case "${profile}" in
    h100)
      if [[ -n "${driver_cuda}" ]] && cropi_version_ge "${driver_cuda}" "12.4"; then
        echo "12.4.0"
      else
        # CUDA 12.2 still supports Hopper/sm_90 and is safer on older 535 drivers.
        echo "12.2.2"
      fi
      ;;
    a100)
      # The observed A100 cluster reports driver CUDA 12.2; keep nvcc at/below it.
      echo "12.2.2"
      ;;
    *)
      if [[ -n "${driver_cuda}" ]] && cropi_version_ge "${driver_cuda}" "12.4"; then
        echo "12.4.0"
      else
        echo "12.2.2"
      fi
      ;;
  esac
}

cropi_apply_vm_compat() {
  local profile
  profile="$(cropi_detect_vm_profile)"
  export CROPI_VM_PROFILE="${profile}"

  case "${profile}" in
    h100)
      export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
      export CROPI_GPU_ARCH_NOTE="${CROPI_GPU_ARCH_NOTE:-H100/Hopper sm_90}"
      ;;
    a100)
      export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
      export CROPI_GPU_ARCH_NOTE="${CROPI_GPU_ARCH_NOTE:-A100/Ampere sm_80}"
      ;;
    *)
      export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
      export CROPI_GPU_ARCH_NOTE="${CROPI_GPU_ARCH_NOTE:-generic CUDA 12.x}"
      ;;
  esac

  export CUDA_REDIST_VER="${CUDA_REDIST_VER:-$(cropi_cuda_redist_default "${profile}")}"

  # Scoring/selection env. Keep it separate from verl; traker/fast_jl are tied to
  # the torch build used here.
  export CROPI_TORCH_SPEC="${CROPI_TORCH_SPEC:-torch==2.4.0}"
  export CROPI_TORCH_INDEX="${CROPI_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
  export TRAKER_SPEC="${TRAKER_SPEC:-traker==0.3.2}"

  # RL env. This is the classic matrix inferred from the recent commits:
  # run_cropi.sh uses verl.trainer.main_ppo, so do not drift to latest verl/vLLM.
  export VERL_TORCH_SPEC="${VERL_TORCH_SPEC:-torch==2.6.0}"
  export VERL_TORCH_INDEX="${VERL_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
  export VLLM_PIP_SPEC="${VLLM_PIP_SPEC:-vllm==0.8.5}"
  export VERL_PIP_SPEC="${VERL_PIP_SPEC:-verl==0.4.1}"
  export TENSORDICT_SPEC="${TENSORDICT_SPEC:-tensordict==0.6.2}"

  # install_venv.sh historically used these shorter names; keep them wired to
  # the same matrix unless the caller explicitly overrides them.
  export TORCH_SPEC="${TORCH_SPEC:-${VERL_TORCH_SPEC}}"
  export TORCH_INDEX="${TORCH_INDEX:-${VERL_TORCH_INDEX}}"
  export VLLM_SPEC="${VLLM_SPEC:-${VLLM_PIP_SPEC}}"

  export CROPI_COMPAT_MATRIX="${CROPI_COMPAT_MATRIX:-classic-verl0.4-vllm0.8-cu124}"
}

cropi_print_vm_compat() {
  cropi_apply_vm_compat
  cat <<EOF
CROPI VM compatibility profile
  profile              = ${CROPI_VM_PROFILE}
  detected driver CUDA = $(cropi_detect_driver_cuda || true)
  gpu arch             = ${TORCH_CUDA_ARCH_LIST} (${CROPI_GPU_ARCH_NOTE})
  CUDA_REDIST_VER      = ${CUDA_REDIST_VER}
  cropi torch          = ${CROPI_TORCH_SPEC} @ ${CROPI_TORCH_INDEX}
  verl torch           = ${VERL_TORCH_SPEC} @ ${VERL_TORCH_INDEX}
  vllm / verl          = ${VLLM_PIP_SPEC} / ${VERL_PIP_SPEC}
  tensordict           = ${TENSORDICT_SPEC}
  matrix               = ${CROPI_COMPAT_MATRIX}
EOF
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "${1:-print}" in
    print) cropi_print_vm_compat ;;
    env)
      cropi_apply_vm_compat
      export -p | grep -E 'CROPI_VM_PROFILE|CROPI_COMPAT_MATRIX|CROPI_TORCH_|TRAKER_SPEC|VERL_|VLLM_|TENSORDICT_SPEC|TORCH_CUDA_ARCH_LIST|CUDA_REDIST_VER|TORCH_SPEC|TORCH_INDEX|VLLM_SPEC'
      ;;
    *) echo "usage: bash scripts/vm_compat.sh [print|env]" >&2; exit 2 ;;
  esac
fi
