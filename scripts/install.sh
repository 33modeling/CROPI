#!/usr/bin/env bash
# Create the two CROPI environments on the workspace disk.
#   bash scripts/install.sh cropi   # scoring + selection env (this repo's code)
#   bash scripts/install.sh verl    # RL + vLLM rollout env (external framework)
#   bash scripts/install.sh all
#
# Run `source scripts/setup_env.sh` FIRST so paths/CROPI_VENV/VERL_VENV are set.
# This never touches the system Python — everything lives under $CROPI_WORK/venvs.
set -euo pipefail
export UV_SYSTEM_CERTS=1
export PATH=/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4

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
uv pip install --system-certs setuptools wheel
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
  # Verified compatible matrix (verl vLLM>=0.8 guide + PyPI metadata, 2026-07):
  #   torch 2.6.0 (cu124) + vllm 0.8.5 (pins torchvision 0.21.0 / torchaudio 2.6.0 /
  #   xformers 0.0.29.post2, transformers>=4.51.1) + verl 0.4.1 + tensordict 0.6.2.
  #   BARE `verl`/`vllm` pull the latest (verl 0.5/0.8 needs a vllm>=0.9 run_headless
  #   async server; latest vllm needs torch 2.8+), which breaks this classic recipe.
  #   Install order matters: torch first, then vllm, then verl, then tensordict.
  #   Docs: https://verl.readthedocs.io/en/latest/README_vllm0.8.html
  local spec="${VERL_PIP_SPEC:-verl==0.4.1}"           # classic ~0.4.x matches vllm 0.8.5
  local vllm_spec="${VLLM_PIP_SPEC:-vllm==0.8.5}"      # pins torch==2.6.0 / xformers 0.0.29.post2
  local torch_spec="${VERL_TORCH_SPEC:-torch==2.6.0}" # torch build vllm 0.8.5 pins
  local torch_index="${VERL_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
  local tensordict_spec="${TENSORDICT_SPEC:-tensordict==0.6.2}"  # vllm>=0.8 ForkingPickler fix
  log "Creating verl env at $VERL_VENV (python ${VERL_PY:-3.11})"
  uv venv "$VERL_VENV" --python "${VERL_PY:-3.11}"
  # shellcheck disable=SC1091
  source "$VERL_VENV/bin/activate"
  log "Installing $torch_spec (cu124) — pin torch before the version-sensitive step"
  uv pip install "$torch_spec" --index-url "$torch_index" || die "torch install failed — set VERL_TORCH_SPEC/VERL_TORCH_INDEX for your CUDA."
  log "Installing $vllm_spec"
  uv pip install "$vllm_spec" || die "vllm install failed — set VLLM_PIP_SPEC to match $torch_spec."
  log "Installing verl ('$spec') — this is the version-sensitive step"
  uv pip install "$spec" || die "verl install failed — set VERL_PIP_SPEC to a version compatible with $vllm_spec."
  log "Installing $tensordict_spec (vllm>=0.8 compatibility)"
  uv pip install "$tensordict_spec" || log "WARN: tensordict pin failed — set TENSORDICT_SPEC to match $spec."
  python -c "import torch,verl; print('verl', getattr(verl,'__version__','?'), '| torch', torch.__version__, '| cuda', torch.version.cuda)" \
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
