#!/usr/bin/env bash
# Re-pin an existing CROPI VM environment after dependency drift.
#
# Usage:
#   bash scripts/repair_vm_libs.sh [a100|h100|generic] [cropi|verl|all]
#
# Examples:
#   source scripts/setup_env_a100.sh 4
#   bash scripts/repair_vm_libs.sh a100 verl
#
#   source scripts/setup_env_h100.sh 8
#   bash scripts/repair_vm_libs.sh h100 all
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

PROFILE=""
WHICH="${1:-all}"
case "${WHICH}" in
  a100|h100|generic)
    PROFILE="${WHICH}"
    shift || true
    WHICH="${1:-all}"
    ;;
esac

case "${PROFILE:-${CROPI_VM_PROFILE:-generic}}" in
  a100)
    # shellcheck disable=SC1091
    source "${HERE}/setup_env_a100.sh" "${CROPI_GPUS:-4}" >/dev/null
    ;;
  h100)
    # shellcheck disable=SC1091
    source "${HERE}/setup_env_h100.sh" "${CROPI_GPUS:-}" >/dev/null
    ;;
  generic|*)
    # shellcheck disable=SC1091
    source "${HERE}/setup_env.sh" >/dev/null
    ;;
esac

# shellcheck disable=SC1091
source "${HERE}/vm_compat.sh"
cropi_apply_vm_compat

log(){ echo "[repair_vm_libs] $*"; }
die(){ echo "[repair_vm_libs][ERROR] $*" >&2; exit 1; }

require_venv_python() {
  local py="$1"
  local label="$2"
  [[ -x "${py}" ]] || die "${label} python not found at ${py}; run the install script first."
}

python_version_report() {
  local py="$1"
  "${py}" - <<'PY'
import importlib
mods = ("torch", "vllm", "verl", "tensordict", "transformers", "traker", "fast_jl")
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"{name}={getattr(mod, '__version__', '?')}")
    except Exception as exc:
        print(f"{name}=MISSING({exc.__class__.__name__}: {exc})")
PY
}

repair_cropi() {
  local py="${CROPI_VENV}/bin/python"
  require_venv_python "${py}" "cropi"
  log "repairing cropi env: ${CROPI_VENV}"
  log "matrix: ${CROPI_VM_PROFILE} ${CROPI_TORCH_SPEC} ${TRAKER_SPEC} arch=${TORCH_CUDA_ARCH_LIST}"

  "${py}" -m pip install --upgrade pip
  "${py}" -m pip install --upgrade --force-reinstall "${CROPI_TORCH_SPEC}" --index-url "${CROPI_TORCH_INDEX}"
  "${py}" -m pip install --upgrade -e "${REPO_ROOT}"
  "${py}" -m pip install --upgrade numpy pandas pyarrow tqdm "transformers<5" math-verify tabulate "huggingface_hub[cli]"
  "${py}" -m pip install --upgrade --force-reinstall --no-deps "${TRAKER_SPEC}"

  if command -v nvcc >/dev/null 2>&1 || [[ -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
    if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
      export PATH="${CUDA_HOME}/bin:${PATH}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    fi
    log "rebuilding fast_jl for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
      "${py}" -m pip install fast_jl --no-build-isolation --no-deps --force-reinstall --no-cache-dir
  else
    log "WARN: no nvcc found; fast_jl not rebuilt."
    log "      Run 'bash scripts/setup_cuda_venv.sh' after this if cropi-get-grad is needed."
  fi

  "${py}" -m compileall -q "${REPO_ROOT}/cropi"
  python_version_report "${py}"
}

repair_verl() {
  local py="${VERL_VENV}/bin/python"
  require_venv_python "${py}" "verl"
  log "repairing verl env: ${VERL_VENV}"
  log "matrix: ${VERL_TORCH_SPEC} -> ${VLLM_PIP_SPEC} -> ${VERL_PIP_SPEC} -> ${TENSORDICT_SPEC}"

  "${py}" -m pip install --upgrade pip
  "${py}" -m pip install --upgrade --force-reinstall "${VERL_TORCH_SPEC}" --index-url "${VERL_TORCH_INDEX}"
  "${py}" -m pip install --upgrade --force-reinstall "${VLLM_PIP_SPEC}"
  "${py}" -m pip install --upgrade --force-reinstall "${VERL_PIP_SPEC}"
  "${py}" -m pip install --upgrade --force-reinstall "${TENSORDICT_SPEC}"

  # Guard against a dependency resolver switching the CUDA-enabled torch wheel.
  "${py}" -m pip install --force-reinstall --no-deps "${VERL_TORCH_SPEC}" --index-url "${VERL_TORCH_INDEX}"

  "${py}" - <<'PY'
import torch
major = (torch.version.cuda or "").split(".")[0]
if major != "12":
    raise SystemExit(f"torch CUDA must be 12.x, got {torch.__version__} cuda={torch.version.cuda}")
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY
  python_version_report "${py}"
}

case "${WHICH}" in
  cropi) repair_cropi ;;
  verl) repair_verl ;;
  all) repair_cropi; repair_verl ;;
  *) die "usage: bash scripts/repair_vm_libs.sh [a100|h100|generic] [cropi|verl|all]" ;;
esac

log "done"
