#!/usr/bin/env bash
# Build fast_jl on a node with NO system CUDA toolkit.
#
# The pip `nvidia-cuda-nvcc-cu12` wheels on this cluster ship only `ptxas` (no
# nvcc frontend — confirmed via pip RECORD), so instead we pull NVIDIA's official
# CUDA *redistributable* tarballs (complete nvcc + cicc + ptxas + nvvm), assemble
# a CUDA_HOME, and build fast_jl against the cropi venv's torch (cu124).
#
#   source scripts/setup_env_a100.sh 4
#   bash scripts/setup_cuda_venv.sh
#
# Override the CUDA version with CUDA_REDIST_VER=12.4.0 etc. Idempotent.
set -euo pipefail

: "${CROPI_VENV:?source scripts/setup_env_a100.sh first}"
: "${CROPI_WORK:?source scripts/setup_env_a100.sh first}"

log(){ echo -e "\n\033[1;36m[cuda] $*\033[0m"; }
die(){ echo "[cuda][ERROR] $*" >&2; exit 1; }

[[ -f "$CROPI_VENV/bin/activate" ]] || die "cropi venv missing at $CROPI_VENV (run scripts/install_venv.sh cropi)"
# shellcheck disable=SC1091
source "$CROPI_VENV/bin/activate"

TORCH_CUDA=$(python -c "import torch;print(torch.version.cuda or '')" 2>/dev/null || true)
log "cropi venv torch CUDA = ${TORCH_CUDA:-unknown} (matching a cu12.x nvcc)"

CUDA_VER="${CUDA_REDIST_VER:-12.4.1}"
BASE="https://developer.download.nvidia.com/compute/cuda/redist"
DL="$CROPI_WORK/venvs/cuda_dl"
CH="$CROPI_WORK/venvs/cuda12_home"
mkdir -p "$DL"

log "fetching NVIDIA redist manifest (redistrib_${CUDA_VER}.json)"
curl -fsSL "$BASE/redistrib_${CUDA_VER}.json" -o "$DL/manifest.json" \
  || die "cannot fetch manifest — try CUDA_REDIST_VER=12.4.0 (nvidia reachable? proxy?)"

# Resolve the exact archive paths for the components we need.
mapfile -t RELS < <(python - "$DL/manifest.json" <<'PY'
import sys, json
m = json.load(open(sys.argv[1]))
for c in ("cuda_nvcc", "cuda_cudart", "cuda_cccl"):
    print(m[c]["linux-x86_64"]["relative_path"])
PY
)
[[ ${#RELS[@]} -eq 3 ]] || die "manifest parse failed (unexpected JSON structure)"

log "downloading + extracting: ${RELS[*]}"
rm -rf "$CH"; mkdir -p "$CH"
for rel in "${RELS[@]}"; do
  f="$DL/$(basename "$rel")"
  [[ -f "$f" ]] || curl -fsSL "$BASE/$rel" -o "$f" || die "download failed: $rel"
  tar -xf "$f" -C "$DL"
done

# Merge each <component>-archive/{bin,include,lib,nvvm} into one CUDA_HOME.
shopt -s nullglob
for d in "$DL"/*-archive; do
  for sub in bin include lib nvvm; do
    [[ -d "$d/$sub" ]] && { mkdir -p "$CH/$sub"; cp -rn "$d/$sub/." "$CH/$sub/" 2>/dev/null || true; }
  done
done
shopt -u nullglob
[[ -d "$CH/lib" && ! -e "$CH/lib64" ]] && ln -sfn "$CH/lib" "$CH/lib64"

export CUDA_HOME="$CH"
export PATH="$CH/bin:$PATH"
export LD_LIBRARY_PATH="$CH/lib64:${LD_LIBRARY_PATH:-}"

log "nvcc check"
[[ -x "$CH/bin/nvcc" ]] || die "nvcc still missing after extract — check $DL/*-archive layout"
nvcc --version
NVCC_MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\).*/\1/p' | head -1)
[[ "$NVCC_MAJOR" == "12" ]] || die "nvcc major=$NVCC_MAJOR, need 12 (set CUDA_REDIST_VER=12.4.x)"

log "building fast_jl against cropi torch"
python -m pip install fast_jl --no-build-isolation --force-reinstall --no-cache-dir \
  || die "fast_jl build failed — paste the compiler error (may need g++)."

log "verify"
python - <<'PY'
import torch, fast_jl, trak
from trak.projectors import CudaProjector, ProjectionType
print("torch", torch.__version__, "cuda", torch.version.cuda)
p = CudaProjector(grad_dim=1024, proj_dim=64, seed=0, proj_type=ProjectionType.normal,
                  device="cuda", dtype=torch.float16, block_size=128, max_batch_size=8)
x = torch.randn(4, 1024, device="cuda", dtype=torch.float16)
print("projection shape:", tuple(p.project(x, model_id=0).shape))
print("fast_jl + trak OK")
PY

log "DONE. CUDA_HOME=$CH"
echo "  setup_env_a100.sh auto-detects this path — future 'source' sets CUDA_HOME."
echo "  cropi env complete -> next: preflight, prep, run_gsm8k."
