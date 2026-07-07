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
  # The node driver caps the CUDA MAJOR you can run (e.g. driver 535 = CUDA 12.2
  # -> only cu12x torch works; a default `pip install verl` drags in a cu13 torch
  # that fails with "driver too old"). So install a cu124 torch FIRST, then vllm,
  # then verl, and finally guard that nothing bumped torch off CUDA 12.
  # Verified compatible matrix (verl vLLM>=0.8 guide + PyPI metadata, 2026-07):
  #   torch 2.6.0 (cu124) + vllm 0.8.5 (pins torchvision 0.21.0 / torchaudio 2.6.0 /
  #   xformers 0.0.29.post2, transformers>=4.51.1) + verl 0.4.1 (classic main_ppo).
  #   BARE `verl` pulls the latest (0.5/0.8) whose main_ppo needs a vllm>=0.9 async
  #   server (run_headless) — that breaks against vllm 0.8.5, so pin verl==0.4.1.
  local verl_spec="${VERL_PIP_SPEC:-verl==0.4.1}"   # classic ~0.4.x matches vllm 0.8.5
  local torch_spec="${TORCH_SPEC:-torch==2.6.0}"    # torch build vllm 0.8.5 pins
  local torch_index="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
  local vllm_spec="${VLLM_SPEC:-vllm==0.8.5}"        # pins torch==2.6.0 / xformers 0.0.29.post2
  local tensordict_spec="${TENSORDICT_SPEC:-tensordict==0.6.2}"  # vllm>=0.8 ForkingPickler fix
  log "Creating verl venv at $VERL_VENV"
  "$PYTHON_BIN" -m venv "$VERL_VENV"
  # shellcheck disable=SC1091
  source "$VERL_VENV/bin/activate"
  python -m pip install --upgrade pip

  log "installing $torch_spec from $torch_index (match the node driver's CUDA major)"
  python -m pip install $torch_spec --index-url "$torch_index" || die "torch install failed"
  log "installing $vllm_spec"
  python -m pip install "$vllm_spec" || die "vllm install failed — adjust VLLM_SPEC to match $torch_spec."
  log "installing verl ('$verl_spec')"
  python -m pip install "$verl_spec" || die "verl install failed — set VERL_PIP_SPEC to a version compatible with $torch_spec."
  # verl>=0.8 path only: main_ppo needs vllm>=0.9 (run_headless). With vllm 0.8.5
  # keep classic verl 0.4.x and pin tensordict==0.6.2, else importing vllm raises
  # "cannot import ForkingPickler from torch.multiprocessing.reductions".
  log "installing $tensordict_spec (vllm>=0.8 compatibility)"
  python -m pip install "$tensordict_spec" || die "tensordict install failed — set TENSORDICT_SPEC to match $verl_spec."

  # guard: if a dep re-pulled a non-cu12 torch, force the cu124 build back
  if ! python -c "import torch,sys; c=torch.version.cuda or ''; sys.exit(0 if c.split('.')[0]=='12' else 1)"; then
    log "a dependency bumped torch off CUDA 12 — reinstalling $torch_spec (cu124)"
    python -m pip install --force-reinstall --no-deps $torch_spec --index-url "$torch_index"
  fi

  python -c "import torch,verl; print('verl', getattr(verl,'__version__','?'), '| torch', torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available())" \
    && log "verl env done -> RL_PYTHON=$VERL_VENV/bin/python (confirm 'avail True' above)" \
    || log "WARN: import failed — check the log; may need different TORCH_SPEC/VLLM_SPEC/VERL_PIP_SPEC."
  deactivate || true
}

case "$WHICH" in
  cropi) install_cropi ;;
  verl)  install_verl ;;
  all)   install_cropi; install_verl ;;
  *) die "usage: bash scripts/install_venv.sh {cropi|verl|all}" ;;
esac

log "Done. Next: 'source scripts/setup_env_a100.sh 4 && cropi_activate', then bash scripts/preflight.sh"
