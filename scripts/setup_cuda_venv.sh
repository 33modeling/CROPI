#!/usr/bin/env bash
# Build fast_jl on a box with NO system CUDA toolkit (containers where only the
# driver is present). Installs a CUDA 12.x toolkit as pip wheels INTO the cropi
# venv, assembles a unified CUDA_HOME out of the split nvidia/* dirs (nvcc needs
# a real toolkit layout), then builds fast_jl against the cropi venv's torch.
#
#   source scripts/setup_env_a100.sh 4
#   bash scripts/setup_cuda_venv.sh
#
# CUDA 12 (not 13) is required: the cropi venv's torch is cu124, and torch's
# CUDAExtension refuses a nvcc whose MAJOR version differs (13 != 12).
# Idempotent: re-running rebuilds the symlink farm and re-checks fast_jl.
set -euo pipefail

: "${CROPI_VENV:?source scripts/setup_env_a100.sh first}"
: "${CROPI_WORK:?source scripts/setup_env_a100.sh first}"

log(){ echo -e "\n\033[1;36m[cuda] $*\033[0m"; }
die(){ echo "[cuda][ERROR] $*" >&2; exit 1; }

[[ -f "$CROPI_VENV/bin/activate" ]] || die "cropi venv missing at $CROPI_VENV (run scripts/install_venv.sh cropi)"
# shellcheck disable=SC1091
source "$CROPI_VENV/bin/activate"

TORCH_CUDA=$(python -c "import torch;print(torch.version.cuda or '')" 2>/dev/null || true)
log "cropi venv torch CUDA = ${TORCH_CUDA:-unknown} (need a cu12 nvcc to match)"

log "installing CUDA 12 build wheels into the cropi venv"
python -m pip install \
  "nvidia-cuda-nvcc-cu12" "nvidia-cuda-runtime-cu12" "nvidia-cuda-cccl-cu12" \
  || die "pip install of cu12 nvcc wheels failed (proxy/index?)."

NV=$(python -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))")
log "nvidia pkg root: $NV"
[[ -x "$NV/cuda_nvcc/bin/nvcc" ]] || die "nvcc not found under $NV/cuda_nvcc/bin (wheel layout changed?)"

# --- assemble a unified CUDA_HOME from the split wheels ----------------------
CH="$CROPI_WORK/venvs/cuda12_home"
log "assembling CUDA_HOME at $CH"
rm -rf "$CH"; mkdir -p "$CH/include" "$CH/lib64"
# nvcc + its private nvvm/bin must keep their relative layout, so link whole dirs
ln -sfn "$NV/cuda_nvcc/bin"  "$CH/bin"
[[ -d "$NV/cuda_nvcc/nvvm" ]] && ln -sfn "$NV/cuda_nvcc/nvvm" "$CH/nvvm"
# merge headers + libs from every wheel that has them
for d in cuda_nvcc cuda_runtime cuda_cccl; do
  [[ -d "$NV/$d/include" ]] && ln -sf "$NV/$d/include/"* "$CH/include/" 2>/dev/null || true
  [[ -d "$NV/$d/lib" ]]     && ln -sf "$NV/$d/lib/"*     "$CH/lib64/"   2>/dev/null || true
done

export CUDA_HOME="$CH"
export PATH="$CH/bin:$PATH"
export LD_LIBRARY_PATH="$CH/lib64:${LD_LIBRARY_PATH:-}"

log "nvcc check"
nvcc --version || die "assembled nvcc is not runnable"
NVCC_MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\).*/\1/p' | head -1)
[[ "$NVCC_MAJOR" == "12" ]] || die "assembled nvcc major=$NVCC_MAJOR, need 12 to match torch cu12x. Pin nvidia-cuda-nvcc-cu12==12.4.* and re-run."

log "building fast_jl against cropi torch"
python -m pip install fast_jl --no-build-isolation --force-reinstall --no-cache-dir \
  || die "fast_jl build failed — paste the compiler error; may need g++ or nvidia-cublas-cu12."

log "verify imports"
python - <<'PY'
import torch, fast_jl, trak
from trak.projectors import CudaProjector, ProjectionType
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("fast_jl OK, trak OK")
# tiny end-to-end projection sanity check on GPU
p = CudaProjector(grad_dim=1024, proj_dim=64, seed=0,
                  proj_type=ProjectionType.normal, device="cuda",
                  dtype=torch.float16, block_size=128, max_batch_size=8)
x = torch.randn(4, 1024, device="cuda", dtype=torch.float16)
print("projection shape:", tuple(p.project(x, model_id=0).shape))
PY

log "DONE. CUDA_HOME=$CH"
echo "  setup_env_a100.sh now auto-detects this path, so future 'source' sets CUDA_HOME for you."
echo "  cropi env is complete -> next: verl env + preflight."
