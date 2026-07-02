#!/usr/bin/env bash
# A100 80G environment for the CROPI gsm8k/mmlu experiments.
#
#   source scripts/setup_env_a100.sh 4     # 4x A100 80G  (default)
#   source scripts/setup_env_a100.sh 8     # 8x A100 80G
#
# Fixed layout for the SR-PAI2026 cluster:
#   datasets : $IT_DATASETS/{gsm8k,mmlu}                         (read-only, shared)
#   model    : /group-volume/SR-PAI2026/nait-models/Qwen3.5-9B   (read-only, shared)
#   workspace: /group-volume/minsoo3.kim/cropi                   (ours; created here)
#
# This sets the A100/9B-specific knobs, then hands off to the repo's setup_env.sh
# (which only fills in ${VAR:-default}, so everything exported here wins).

# ---- GPU count (positional arg or $CROPI_GPUS; default 4) --------------------
_CROPI_GPUS="${1:-${CROPI_GPUS:-4}}"
if [[ "${_CROPI_GPUS}" != "4" && "${_CROPI_GPUS}" != "8" ]]; then
  echo "[setup_env_a100] usage: source scripts/setup_env_a100.sh [4|8]  (got '${_CROPI_GPUS}')" >&2
  return 1 2>/dev/null || exit 1
fi
export CROPI_GPUS="${_CROPI_GPUS}"

# ---- Cluster paths (override before sourcing if a mount differs) -------------
export GROUP_VOLUME="${GROUP_VOLUME:-/group-volume}"
export IT_DATASETS="${IT_DATASETS:-${GROUP_VOLUME}/SR-PAI2026/IT-datasets}"

# All heavy artefacts (data, rollouts, grads, checkpoints, logs) go here.
export CROPI_WORK="${CROPI_WORK:-${GROUP_VOLUME}/minsoo3.kim/cropi}"
export DATA_ROOT="${DATA_ROOT:-${CROPI_WORK}/data}"
export CKPT_ROOT="${CKPT_ROOT:-${CROPI_WORK}/checkpoints}"
export RESULTS_DIR="${RESULTS_DIR:-${CROPI_WORK}/results}"

# Base model: the shared Qwen3.5-9B checkout (read-only).
export MODELS_DIR="${MODELS_DIR:-${GROUP_VOLUME}/nait-models}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODELS_DIR}/Qwen3.5-9B}"
export HFID_BASE_MODEL="${HFID_BASE_MODEL:-Qwen/Qwen3.5-9B}"   # only used if you re-download
# Exact name varies (case / .5 / -Instruct); if the default path is absent,
# best-effort auto-detect a Qwen3*9B dir under nait-models.
if [[ ! -d "${BASE_MODEL_PATH}" ]]; then
  _bm=""
  for _root in "${MODELS_DIR}" "${GROUP_VOLUME}/nait-models" "${GROUP_VOLUME}/SR-PAI2026/nait-models"; do
    [[ -d "${_root}" ]] || continue
    _bm=$(find "${_root}" -maxdepth 1 -type d -iname '*qwen3*9b*' 2>/dev/null | sort | head -1)
    [[ -n "${_bm}" ]] && break
  done
  if [[ -n "${_bm}" ]]; then
    export BASE_MODEL_PATH="${_bm}"
    echo "[setup_env_a100] auto-detected BASE_MODEL_PATH=${BASE_MODEL_PATH}"
  fi
fi

# ---- Hardware-dependent knobs (MUST match visible GPU count) -----------------
# cropi-get-grad pins shard k to gpu = k % NUM_PARALLEL, so NUM_PARALLEL must
# equal the number of visible GPUs. RL uses the same count for its FSDP world.
export NUM_PARALLEL="${NUM_PARALLEL:-${CROPI_GPUS}}"
export RL_NUM_GPUS="${RL_NUM_GPUS:-${CROPI_GPUS}}"
export RL_TP_SIZE="${RL_TP_SIZE:-2}"          # 9B: vLLM tensor-parallel across 2 GPUs

# ---- 9B on 80G: relaxed vs the 4090 fork defaults, still conservative --------
export RL_GPU_MEMORY_UTILIZATION="${RL_GPU_MEMORY_UTILIZATION:-0.6}"
export RL_PPO_MICRO_BATCH_SIZE_PER_GPU="${RL_PPO_MICRO_BATCH_SIZE_PER_GPU:-8}"
export RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}"

# ---- Experiment defaults (gsm8k first; shared by both arms) ------------------
export PROMPT_TYPE="${PROMPT_TYPE:-qwen25-math-cot}"
export TEMPERATURE="${TEMPERATURE:-0.5}"
export N_SAMPLES="${N_SAMPLES:-8}"            # rollouts per train prompt
export N_SAMPLES_VAL="${N_SAMPLES_VAL:-8}"    # rollouts per valid prompt (paper 32; 8 for 9B memory)
export SEED="${SEED:-0}"
export SELECT_RATIO="${SELECT_RATIO:-0.1}"    # CROPI keeps 10%
export SCORE_METHOD="${SCORE_METHOD:-inf_valid_uniform}"
export NUM_RL_ROUNDS="${NUM_RL_ROUNDS:-3}"    # 3-round curriculum for the CROPI arm
export PROJECTION_METHOD="${PROJECTION_METHOD:-trak_norm}"
export PROJ_DIM="${PROJ_DIM:-32768}"
export SPARSE_DIM="${SPARSE_DIM:-15000000}"

# Response/prompt lengths. NOTE: if Qwen3.5-9B is a *thinking* model, raise
# RL_MAX_RESPONSE_LENGTH (preflight will tell you) — 2048 truncates reasoning.
export RL_MAX_PROMPT_LENGTH="${RL_MAX_PROMPT_LENGTH:-1024}"
export RL_MAX_RESPONSE_LENGTH="${RL_MAX_RESPONSE_LENGTH:-2048}"

# Same total training steps for BOTH arms (fair "data efficiency" comparison).
export RL_TOTAL_TRAINING_STEPS="${RL_TOTAL_TRAINING_STEPS:-60}"
export RL_TRAIN_BATCH_SIZE="${RL_TRAIN_BATCH_SIZE:-128}"
export RL_PPO_MINI_BATCH_SIZE="${RL_PPO_MINI_BATCH_SIZE:-128}"

# ---- CUDA toolkit for building fast_jl (needs nvcc, not just torch's runtime) --
# fast_jl compiles a CUDA extension via torch.utils.cpp_extension, which requires
# CUDA_HOME -> a real toolkit with nvcc. Best-effort autodetect; warn if missing.
if [[ -z "${CUDA_HOME:-}" ]]; then
  # Prefer a real system toolkit; else the pip-assembled one from setup_cuda_venv.sh.
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
  echo "[setup_env_a100] CUDA_HOME=${CUDA_HOME} ($("${CUDA_HOME}/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release //p'))"
else
  echo "[setup_env_a100] WARN: no nvcc/CUDA toolkit found -> fast_jl build will fail."
  echo "                 'module load cuda/12.x' or set CUDA_HOME before sourcing (see README/SETUP)."
fi

# Hand off to the base setup (fills venv paths, HF cache, cropi_activate, warnings).
_A100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck disable=SC1091
source "${_A100_DIR}/setup_env.sh"

echo ""
echo "[setup_env_a100] A100 profile active: ${CROPI_GPUS} GPU(s)"
echo "  NUM_PARALLEL=${NUM_PARALLEL}  RL_NUM_GPUS=${RL_NUM_GPUS}  RL_TP_SIZE=${RL_TP_SIZE}"
echo "  BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "  IT_DATASETS=${IT_DATASETS}"
echo "  DATA_ROOT=${DATA_ROOT}"
echo "  CKPT_ROOT=${CKPT_ROOT}"
echo "  RESULTS_DIR=${RESULTS_DIR}"
echo "  NUM_RL_ROUNDS=${NUM_RL_ROUNDS}  SELECT_RATIO=${SELECT_RATIO}  TOTAL_STEPS=${RL_TOTAL_TRAINING_STEPS}"
