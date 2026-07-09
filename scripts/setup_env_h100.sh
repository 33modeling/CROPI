#!/usr/bin/env bash
# H100/Hopper environment profile for CROPI experiments.
#
#   source scripts/setup_env_h100.sh        # auto-detect GPU count when possible
#   source scripts/setup_env_h100.sh 8      # explicit 1/2/4/8 GPUs
#
# This mirrors setup_env_a100.sh but keeps a separate workspace and builds CUDA
# extensions for sm_90. It still uses the classic verl/vLLM matrix from the
# recent commits because cropi/scripts/run_cropi.sh calls verl.trainer.main_ppo.

_H100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_CROPI_PREV_PROFILE="${CROPI_VM_PROFILE:-}"
export CROPI_VM_PROFILE="h100"
if [[ -n "${_CROPI_PREV_PROFILE}" && "${_CROPI_PREV_PROFILE}" != "${CROPI_VM_PROFILE}" ]]; then
  unset TORCH_CUDA_ARCH_LIST CROPI_GPU_ARCH_NOTE CUDA_REDIST_VER
  unset CROPI_WORK DATA_ROOT CKPT_ROOT RESULTS_DIR
  unset NUM_PARALLEL RL_NUM_GPUS RL_TP_SIZE RL_GPU_MEMORY_UTILIZATION
  unset RL_PPO_MICRO_BATCH_SIZE_PER_GPU RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU
fi
# shellcheck disable=SC1091
source "${_H100_DIR}/vm_compat.sh"
cropi_apply_vm_compat

if [[ -z "${1:-}" && -z "${CROPI_GPUS:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  _CROPI_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')"
else
  _CROPI_GPUS="${1:-${CROPI_GPUS:-8}}"
fi
case "${_CROPI_GPUS}" in
  1|2|4|8) ;;
  *) echo "[setup_env_h100] usage: source scripts/setup_env_h100.sh [1|2|4|8]  (got '${_CROPI_GPUS}')" >&2
     return 1 2>/dev/null || exit 1 ;;
esac
export CROPI_GPUS="${_CROPI_GPUS}"

# Cluster/plain-VM paths. Override before sourcing if mounts differ.
export GROUP_VOLUME="${GROUP_VOLUME:-/group-volume}"
export CROPI_USER="${CROPI_USER:-${USER:-$(whoami)}}"
export IT_DATASETS="${IT_DATASETS:-${GROUP_VOLUME}/SR-PAI2026/IT-datasets}"
if [[ ! -d "${IT_DATASETS}/gsm8k" ]]; then
  for _r in "${GROUP_VOLUME}/IT-datasets" "${GROUP_VOLUME}/SR-PAI2026/IT-datasets" \
            "${GROUP_VOLUME}/datasets" "${GROUP_VOLUME}"; do
    [[ -d "${_r}/gsm8k" ]] && export IT_DATASETS="${_r}" && break
  done
  if [[ ! -d "${IT_DATASETS}/gsm8k" ]]; then
    _ds=$(find "${GROUP_VOLUME}" -maxdepth 3 -type d -iname gsm8k 2>/dev/null | head -1)
    [[ -n "${_ds}" ]] && export IT_DATASETS="$(dirname "${_ds}")"
  fi
  [[ -d "${IT_DATASETS}/gsm8k" ]] && echo "[setup_env_h100] auto-detected IT_DATASETS=${IT_DATASETS}"
fi

# Keep H100 venv/checkpoints separate from A100 because fast_jl is built for sm_90.
export CROPI_WORK="${CROPI_WORK:-${GROUP_VOLUME}/${CROPI_USER}/cropi-h100}"
export DATA_ROOT="${DATA_ROOT:-${CROPI_WORK}/data}"
export CKPT_ROOT="${CKPT_ROOT:-${CROPI_WORK}/checkpoints}"
export RESULTS_DIR="${RESULTS_DIR:-${CROPI_WORK}/results}"

export MODELS_DIR="${MODELS_DIR:-${GROUP_VOLUME}/nait-models}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODELS_DIR}/Qwen2.5-7B-Instruct}"
export HFID_BASE_MODEL="${HFID_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_GLOB="${MODEL_GLOB:-*qwen2.5*7b*}"
if [[ ! -d "${BASE_MODEL_PATH}" ]]; then
  _bm=""
  for _root in "${MODELS_DIR}" "${GROUP_VOLUME}/nait-models" "${GROUP_VOLUME}/SR-PAI2026/nait-models"; do
    [[ -d "${_root}" ]] || continue
    _bm=$(find "${_root}" -maxdepth 1 -type d -iname "${MODEL_GLOB}" 2>/dev/null | sort | head -1)
    [[ -n "${_bm}" ]] && break
  done
  if [[ -n "${_bm}" ]]; then
    export BASE_MODEL_PATH="${_bm}"
    echo "[setup_env_h100] auto-detected BASE_MODEL_PATH=${BASE_MODEL_PATH}"
  fi
fi

# Hardware knobs. H100 80G generally handles larger batches than A100/4090, but
# keep these conservative enough for 7B/9B and raise after watching nvidia-smi.
export NUM_PARALLEL="${NUM_PARALLEL:-${CROPI_GPUS}}"
export RL_NUM_GPUS="${RL_NUM_GPUS:-${CROPI_GPUS}}"
export RL_TP_SIZE="${RL_TP_SIZE:-1}"
export RL_GPU_MEMORY_UTILIZATION="${RL_GPU_MEMORY_UTILIZATION:-0.72}"
export RL_PPO_MICRO_BATCH_SIZE_PER_GPU="${RL_PPO_MICRO_BATCH_SIZE_PER_GPU:-16}"
export RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}"

export PROMPT_TYPE="${PROMPT_TYPE:-qwen25-math-cot}"
export TEMPERATURE="${TEMPERATURE:-0.5}"
export N_SAMPLES="${N_SAMPLES:-8}"
export N_SAMPLES_VAL="${N_SAMPLES_VAL:-8}"
export SEED="${SEED:-0}"
export SELECT_RATIO="${SELECT_RATIO:-0.1}"
export SCORE_METHOD="${SCORE_METHOD:-inf_valid_uniform}"
export NUM_RL_ROUNDS="${NUM_RL_ROUNDS:-3}"
export PROJECTION_METHOD="${PROJECTION_METHOD:-trak_norm}"
export PROJ_DIM="${PROJ_DIM:-32768}"
export SPARSE_DIM="${SPARSE_DIM:-15000000}"

export RL_MAX_PROMPT_LENGTH="${RL_MAX_PROMPT_LENGTH:-1024}"
export RL_MAX_RESPONSE_LENGTH="${RL_MAX_RESPONSE_LENGTH:-2048}"
export RL_TOTAL_TRAINING_STEPS="${RL_TOTAL_TRAINING_STEPS:-60}"
export RL_TRAIN_BATCH_SIZE="${RL_TRAIN_BATCH_SIZE:-128}"
export RL_PPO_MINI_BATCH_SIZE="${RL_PPO_MINI_BATCH_SIZE:-128}"

# Prefer a real toolkit; else setup_cuda_venv.sh can assemble one under CROPI_WORK.
if [[ -z "${CUDA_HOME:-}" ]]; then
  for _c in /usr/local/cuda /usr/local/cuda-12.[0-9] /usr/local/cuda-12.[0-9][0-9] \
            "${CROPI_WORK}/venvs/cuda12_home"; do
    [[ -x "${_c}/bin/nvcc" ]] && CUDA_HOME="${_c}" && break
  done
  if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi
fi
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  export CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  echo "[setup_env_h100] CUDA_HOME=${CUDA_HOME} ($("${CUDA_HOME}/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release //p'))"
else
  echo "[setup_env_h100] WARN: no nvcc/CUDA toolkit found -> fast_jl build will need scripts/setup_cuda_venv.sh."
fi

# Hand off to the base setup (fills venv paths, HF cache, cropi_activate, warnings).
# shellcheck disable=SC1091
source "${_H100_DIR}/setup_env.sh"

echo ""
echo "[setup_env_h100] H100 profile active: ${CROPI_GPUS} GPU(s)"
echo "  NUM_PARALLEL=${NUM_PARALLEL}  RL_NUM_GPUS=${RL_NUM_GPUS}  RL_TP_SIZE=${RL_TP_SIZE}"
echo "  BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "  IT_DATASETS=${IT_DATASETS}"
echo "  DATA_ROOT=${DATA_ROOT}"
echo "  CKPT_ROOT=${CKPT_ROOT}"
echo "  RESULTS_DIR=${RESULTS_DIR}"
echo "  VM_PROFILE=${CROPI_VM_PROFILE}  MATRIX=${CROPI_COMPAT_MATRIX}"
echo "  CUDA_REDIST_VER=${CUDA_REDIST_VER}  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "  NUM_RL_ROUNDS=${NUM_RL_ROUNDS}  SELECT_RATIO=${SELECT_RATIO}  TOTAL_STEPS=${RL_TOTAL_TRAINING_STEPS}"

unset _CROPI_PREV_PROFILE
