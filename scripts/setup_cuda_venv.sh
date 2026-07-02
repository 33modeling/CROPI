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

# Match (or stay below) the NODE DRIVER's CUDA version, else the driver can't JIT
# the generated PTX ("unsupported toolchain"). Driver here = CUDA 12.2 -> nvcc 12.2.
CUDA_VER="${CUDA_REDIST_VER:-12.2.2}"
BASE="https://developer.download.nvidia.com/compute/cuda/redist"
DL="$CROPI_WORK/venvs/cuda_dl"
CH="$CROPI_WORK/venvs/cuda12_home"
# fresh each run so stale extracted archives (e.g. cuda_cccl) don't leak in
rm -rf "$DL" "$CH"; mkdir -p "$DL" "$CH"

log "fetching NVIDIA redist manifest (redistrib_${CUDA_VER}.json)"
curl -fsSL "$BASE/redistrib_${CUDA_VER}.json" -o "$DL/manifest.json" \
  || die "cannot fetch manifest — try CUDA_REDIST_VER=12.4.0 (nvidia reachable? proxy?)"

# Only nvcc + cudart. NOT cuda_cccl: its libcu++ <cstdlib>/<cmath> shadow the
# system C++ headers on the -I path and break the compile; fast_jl needs neither.
mapfile -t RELS < <(python - "$DL/manifest.json" <<'PY'
import sys, json
m = json.load(open(sys.argv[1]))
for c in ("cuda_nvcc", "cuda_cudart"):
    print(m[c]["linux-x86_64"]["relative_path"])
PY
)
[[ ${#RELS[@]} -eq 2 ]] || die "manifest parse failed (unexpected JSON structure)"

log "downloading + extracting: ${RELS[*]}"
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

# torch's ATen headers also #include cusparse.h / cublas_v2.h / etc. Those headers
# (and libs) are already in the cropi venv as pip nvidia-* component wheels — merge
# their include/ and lib/ into CUDA_HOME so the fast_jl compile finds them.
NV=$(python -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))" 2>/dev/null || true)
if [[ -n "$NV" && -d "$NV" ]]; then
  log "merging headers/libs from pip nvidia components ($NV)"
  mkdir -p "$CH/include" "$CH/lib"
  # skip cuda_cccl here (its bare libcu++ <cstdlib>/<cmath> shadow the system headers)
  while IFS= read -r inc; do cp -rn "$inc/." "$CH/include/" 2>/dev/null || true; done \
    < <(find "$NV" -type d -name include -not -path '*cccl*' 2>/dev/null)
  # ...but cudart's cuda_fp16.h needs <nv/target>, so bring in ONLY cccl's nv/ and
  # cuda/ subtrees (namespaced — no shadowing), not the bare-name libcu++ headers.
  # namespaced subtrees only (<nv/target>, <thrust/complex.h>, <cub/...>, <cuda/std/...>)
  # — torch's c10 headers include these; the bare libcu++ <cstdlib>/<cmath> stay out.
  _cccl=$(find "$NV" -type d -path '*cccl*' -name include 2>/dev/null | head -1)
  if [[ -n "$_cccl" ]]; then
    for sub in nv cuda thrust cub; do
      [[ -d "$_cccl/$sub" ]] && cp -rn "$_cccl/$sub" "$CH/include/" 2>/dev/null || true
    done
  fi
  while IFS= read -r so; do ln -sf "$so" "$CH/lib/" 2>/dev/null || true; done \
    < <(find "$NV" -type f -name '*.so*' 2>/dev/null)
fi
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
# Build NATIVE for the GPU arch (A100 = sm_80) so no runtime PTX JIT is needed;
# override TORCH_CUDA_ARCH_LIST for other GPUs (e.g. 8.6 for A6000, 9.0 for H100).
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
# --no-deps is CRITICAL: without it, --force-reinstall cascades to reinstalling
# fast_jl's torch dep and silently upgrades torch (breaking the cu124/abi match).
_bl="$CROPI_WORK/venvs/fast_jl_build.log"
if ! python -m pip install fast_jl --no-build-isolation --no-deps --force-reinstall --no-cache-dir > "$_bl" 2>&1; then
  echo "----- real compiler error (from $_bl) -----" >&2
  grep -iE 'fatal error|error:|No such file|undefined reference|FAILED:|ninja: build stopped' "$_bl" | head -25 >&2
  die "fast_jl build failed — see $_bl for the full log."
fi

log "verify"
python - <<'PY'
import torch, fast_jl, trak
from trak.projectors import CudaProjector, ProjectionType
print("torch", torch.__version__, "cuda", torch.version.cuda)
p = CudaProjector(grad_dim=4096, proj_dim=512, seed=0, proj_type=ProjectionType.normal,
                  device="cuda", dtype=torch.float16, block_size=128, max_batch_size=8)  # proj_dim must be a multiple of 512
x = torch.randn(4, 4096, device="cuda", dtype=torch.float16)
print("projection shape:", tuple(p.project(x, model_id=0).shape))
print("fast_jl + trak OK")
PY

log "DONE. CUDA_HOME=$CH"
echo "  setup_env_a100.sh auto-detects this path — future 'source' sets CUDA_HOME."
echo "  cropi env complete -> next: preflight, prep, run_gsm8k."
