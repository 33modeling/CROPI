#!/usr/bin/env bash
# Create the two CROPI environments on the workspace disk.
#   bash scripts/install.sh cropi   # scoring + selection env (this repo's code)
#   bash scripts/install.sh verl    # RL + vLLM rollout env (external framework)
#   bash scripts/install.sh all
#
# Run `source scripts/setup_env.sh` FIRST so paths/CROPI_VENV/VERL_VENV are set.
# This never touches the system Python — everything lives under $CROPI_WORK/venvs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

# Make sure the env is loaded (idempotent if already sourced).
# shellcheck disable=SC1091
source "${HERE}/setup_env.sh"

WHICH="${1:-all}"

log()  { echo "[install] $*"; }
die()  { echo "[install][ERROR] $*" >&2; exit 1; }

bootstrap_uv() {
  if command -v uv >/dev/null 2>&1; then
    log "uv present: $(uv --version)"
    return
  fi

  log "uv not found — installing to ~/.local/bin (no root needed)"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh || log "WARN: uv shell installer failed; trying pip fallback"
  else
    log "WARN: curl not found; trying pip fallback for uv"
  fi
  export PATH="$HOME/.local/bin:$PATH"

  if ! command -v uv >/dev/null 2>&1; then
    command -v python3 >/dev/null 2>&1 || die "uv install failed and python3 is unavailable. Install curl or python3-pip, then re-run."
    python3 -m pip install --user --upgrade uv || log "WARN: pip fallback for uv failed"
    export PATH="$HOME/.local/bin:$PATH"
  fi

  command -v uv >/dev/null 2>&1 || die "uv install failed; install it manually: https://docs.astral.sh/uv/"
  log "uv installed: $(uv --version)"
}

require_workspace() {
  [[ "${CROPI_WORK_WRITABLE:-0}" == "1" ]] \
    || die "CROPI_WORK is not writable: $CROPI_WORK. Export CROPI_WORK or GROUP_VOLUME to a writable workspace and re-run."
  mkdir -p "$(dirname "$CROPI_VENV")" "$(dirname "$VERL_VENV")"
}

install_cropi() {
  log "Creating cropi env at $CROPI_VENV (python $CROPI_PY)"
  uv venv "$CROPI_VENV" --python "$CROPI_PY"
  # shellcheck disable=SC1091
  source "$CROPI_VENV/bin/activate"

  # Exact recipe from the upstream README (§ How To Run / 1).
  log "Installing torch 2.4.0 (cu124) + cropi + selection deps"
  uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
  uv pip install -e "$REPO_ROOT"
  uv pip install numpy pandas pyarrow tqdm "transformers<5" math-verify tabulate "huggingface_hub[cli]"
  uv pip install --no-deps traker==0.3.2
  # fast_jl builds a CUDA extension against the env's torch — needs a matching
  # CUDA toolkit (nvcc) on PATH. Most GPU cloud images ship it; if this step
  # fails, install the CUDA 12.x toolkit (or `module load cuda`) and re-run.
  uv pip install fast_jl --no-build-isolation \
    || die "fast_jl build failed — likely no nvcc/CUDA toolkit. Install CUDA 12.x toolkit and re-run 'bash scripts/install.sh cropi'."

  log "cropi env sanity check"
  uv run --python "$CROPI_VENV/bin/python" python -m compileall -q "$REPO_ROOT/cropi" \
    && log "cropi env OK ✓"
  deactivate || true
}

install_verl() {
  # verl is a heavy, version-coupled framework (torch + vLLM + flash-attn + ray).
  # We create an isolated venv and do a best-effort install; pin/adjust to match
  # your CUDA per the verl docs: https://verl.readthedocs.io/en/latest/start/install.html
  local spec="${VERL_PIP_SPEC:-verl}"          # e.g. VERL_PIP_SPEC='verl==0.4.1'
  local vllm_spec="${VLLM_PIP_SPEC:-vllm}"     # e.g. VLLM_PIP_SPEC='vllm==0.8.5'
  log "Creating verl env at $VERL_VENV (python ${VERL_PY:-3.11})"
  uv venv "$VERL_VENV" --python "${VERL_PY:-3.11}"
  # shellcheck disable=SC1091
  source "$VERL_VENV/bin/activate"
  log "Installing verl ('$spec') + vllm ('$vllm_spec') — this is the version-sensitive step"
  uv pip install "$spec" || die "verl install failed — follow the verl install guide and set VERL_PIP_SPEC, then re-run."
  # vLLM is what verl uses for rollout; install if the verl meta-package didn't pull it.
  python -c "import vllm" 2>/dev/null || uv pip install "$vllm_spec" || log "WARN: vllm not installed — install a verl-compatible vllm manually."
  python -c "import verl; print('verl', getattr(verl,'__version__','?'))" \
    && log "verl env OK ✓  ->  RL_PYTHON=$VERL_VENV/bin/python" \
    || log "WARN: 'import verl' failed — check the install log."
  deactivate || true
}

require_workspace
bootstrap_uv
case "$WHICH" in
  cropi) install_cropi ;;
  verl)  install_verl ;;
  all)   install_cropi; install_verl ;;
  *) die "usage: bash scripts/install.sh {cropi|verl|all}" ;;
esac

log "Done. Next: 'source scripts/setup_env.sh && cropi_activate' then see SETUP.md."
