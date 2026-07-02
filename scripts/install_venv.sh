#!/usr/bin/env bash
# venv-based install (NO uv) — for machines where the uv bootstrap is blocked
# (e.g. corporate net returns 403 on astral.sh). Uses the system python's venv
# + pip, which already works through the configured proxy / internal mirror.
#
#   source scripts/setup_env_a100.sh 4    # sets CROPI_VENV / VERL_VENV / paths
#   bash scripts/install_venv.sh cropi    # scoring + selection env
#   bash scripts/install_venv.sh verl     # RL + vLLM env
#   bash scripts/install_venv.sh all
#
# Override the interpreter with PYTHON_BIN=python3.11 (needs >=3.10).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
# shellcheck disable=SC1091
source "${HERE}/setup_env.sh" >/dev/null

WHICH="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log(){ echo "[install_venv] $*"; }
die(){ echo "[install_venv][ERROR] $*" >&2; exit 1; }

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "$PYTHON_BIN not found; set PYTHON_BIN=python3.x"
log "using $($PYTHON_BIN --version) at $(command -v "$PYTHON_BIN")"

install_cropi() {
  log "Creating cropi venv at $CROPI_VENV"
  "$PYTHON_BIN" -m venv "$CROPI_VENV"
  # shellcheck disable=SC1091
  source "$CROPI_VENV/bin/activate"
  python -m pip install --upgrade pip

  # torch first, from the CUDA 12.4 wheel index. If download.pytorch.org is
  # blocked (403), install torch from your internal mirror instead, then re-run
  # this script (the torch line will be a no-op).
  log "installing torch 2.4.0 (cu124)"
  python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 \
    || die "torch install failed (proxy/403?). Install torch==2.4.0 from your internal mirror, then re-run."

  log "installing cropi + selection deps"
  python -m pip install -e "$REPO_ROOT"
  python -m pip install numpy pandas pyarrow tqdm "transformers<5" math-verify tabulate
  python -m pip install --no-deps traker==0.3.2
  # fast_jl compiles a CUDA extension against this env's torch -> needs nvcc (CUDA 12.x).
  python -m pip install fast_jl --no-build-isolation \
    || die "fast_jl build failed — likely no nvcc/CUDA toolkit on PATH. Load CUDA 12.x and re-run 'bash scripts/install_venv.sh cropi'."

  python -m compileall -q "$REPO_ROOT/cropi" && log "cropi env OK ✓ ($CROPI_VENV)"
  deactivate || true
}

install_verl() {
  local spec="${VERL_PIP_SPEC:-verl}"      # e.g. VERL_PIP_SPEC='verl==0.4.1'
  log "Creating verl venv at $VERL_VENV"
  "$PYTHON_BIN" -m venv "$VERL_VENV"
  # shellcheck disable=SC1091
  source "$VERL_VENV/bin/activate"
  python -m pip install --upgrade pip
  log "installing verl ('$spec') + vllm — version-sensitive step"
  python -m pip install "$spec" || die "verl install failed — set VERL_PIP_SPEC to a CUDA-matched version and re-run."
  python -c "import vllm" 2>/dev/null || python -m pip install vllm || log "WARN: vllm not installed — install a verl-compatible vllm manually."
  python -c "import verl; print('verl', getattr(verl,'__version__','?'))" \
    && log "verl env OK ✓  ->  RL_PYTHON=$VERL_VENV/bin/python" \
    || log "WARN: 'import verl' failed — check the log above."
  deactivate || true
}

case "$WHICH" in
  cropi) install_cropi ;;
  verl)  install_verl ;;
  all)   install_cropi; install_verl ;;
  *) die "usage: bash scripts/install_venv.sh {cropi|verl|all}" ;;
esac

log "Done. Next: 'source scripts/setup_env_a100.sh 4 && cropi_activate', then bash scripts/preflight.sh"
