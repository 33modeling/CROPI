#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/minsoo3.kim/dev/CROPI"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

export GROUP_VOLUME="${GROUP_VOLUME:-/data1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-/data1/minsoo3.kim/weasel/models/Qwen2.5-3B-Instruct}"
export TRAIN_DATA_NAMES="${TRAIN_DATA_NAMES:-gsm8k}"
export VALID_DATA_NAMES="${VALID_DATA_NAMES:-gsm8k}"
export RL_VAL_DATA_NAMES="${RL_VAL_DATA_NAMES:-gsm8k}"
export PROMPT_TYPE="${PROMPT_TYPE:-qwen25-math-cot}"
export SEED="${SEED:-0}"
export TEMPERATURE="${TEMPERATURE:-0.5}"
export N_SAMPLES="${N_SAMPLES:-4}"
export N_SAMPLES_VAL="${N_SAMPLES_VAL:-4}"
export RL_MAX_RESPONSE_LENGTH="${RL_MAX_RESPONSE_LENGTH:-1024}"
export RL_MAX_PROMPT_LENGTH="${RL_MAX_PROMPT_LENGTH:-1024}"
export RL_TP_SIZE="${RL_TP_SIZE:-2}"
export RL_NUM_GPUS="${RL_NUM_GPUS:-2}"
export RL_GPU_MEMORY_UTILIZATION="${RL_GPU_MEMORY_UTILIZATION:-0.5}"
export NUM_PARALLEL="${NUM_PARALLEL:-2}"
export SELECT_RATIO="${SELECT_RATIO:-0.1}"
export RL_TRAIN_BATCH_SIZE="${RL_TRAIN_BATCH_SIZE:-32}"
export RL_PPO_MINI_BATCH_SIZE="${RL_PPO_MINI_BATCH_SIZE:-32}"
export RL_PPO_MICRO_BATCH_SIZE_PER_GPU="${RL_PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
export RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
export RL_TOTAL_TRAINING_STEPS="${RL_TOTAL_TRAINING_STEPS:-150}"
export RL_SAVE_FREQ="${RL_SAVE_FREQ:-150}"
export RL_TEST_FREQ="${RL_TEST_FREQ:-50}"
export NUM_RL_ROUNDS="${NUM_RL_ROUNDS:-1}"
export CUSTOM_REWARD_PATH="${CUSTOM_REWARD_PATH:-${REPO_ROOT}/cropi/rewards/gsm8k_math.py}"
export CUSTOM_REWARD_NAME="${CUSTOM_REWARD_NAME:-compute_score}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CROPI_RUN="${CROPI_RUN:-}"
export PATH="/data1/minsoo3.kim/cropi/venvs/cropi/bin:${PATH}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/setup_env.sh" >/dev/null 2>&1

MODEL_NAME="${MODEL_NAME:-Qwen2.5-3B_curriculum}"
INFER_NOTE="${PROMPT_TYPE}_-1_seed${SEED}_t${TEMPERATURE}_n${N_SAMPLES}_s0_e-1"
VALID_INFER_NOTE="${PROMPT_TYPE}_-1_seed${SEED}_t${TEMPERATURE}_n${N_SAMPLES_VAL}_s0_e-1"
PROJ_NOTE="${PROJ_NOTE:-trak_norm_seed${SEED}_mid0_projdim32768_sparse15000000}"

RESULTS_DIR="${RESULTS_DIR:-${CROPI_WORK}/results}"
LOG_ROOT="${LOG_ROOT:-${CROPI_WORK}/logs/gsm8k_ab_${RUN_ID}}"
REPORT_HTML="${REPORT_HTML:-${RESULTS_DIR}/gsm8k_ab_report_${RUN_ID}.html}"
TIMINGS_TSV="${LOG_ROOT}/timings.tsv"
mkdir -p "${RESULTS_DIR}" "${LOG_ROOT}" "${DATA_ROOT}/gsm8k/${MODEL_NAME}"

A_CKPT_ROOT="${A_CKPT_ROOT:-${CKPT_ROOT}/cropi_rl/gsm8k_full_3b}"
A_CKPT="${A_CKPT:-${A_CKPT_ROOT}/global_step_150/actor/huggingface}"
B_CKPT_ROOT="${B_CKPT_ROOT:-${CKPT_ROOT}/cropi_rl/iter0}"
B_CKPT="${B_CKPT:-${B_CKPT_ROOT}/global_step_${RL_TOTAL_TRAINING_STEPS}/actor/huggingface}"
EVAL_PARQUET="${EVAL_PARQUET:-${DATA_ROOT}/gsm8k/valid_qwen.parquet}"
A_RESULT="${A_RESULT:-${RESULTS_DIR}/gsm8k_armA_full_3b_step150.json}"
B_RESULT="${B_RESULT:-${RESULTS_DIR}/gsm8k_armB_cropi10_3b_step${RL_TOTAL_TRAINING_STEPS}.json}"

TRAIN_JSONL="${DATA_ROOT}/gsm8k/${MODEL_NAME}/train_${INFER_NOTE}.jsonl"
VALID_JSONL="${DATA_ROOT}/gsm8k/${MODEL_NAME}/valid_${VALID_INFER_NOTE}.jsonl"
POOL="${POOL:-2000}"
VPOOL="${VPOOL:-256}"

echo -e "phase\tstatus\tstart_iso\tend_iso\tseconds\tlog" > "${TIMINGS_TSV}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

have_hf_model() {
  local dir="$1"
  [[ -f "${dir}/config.json" && -f "${dir}/model.safetensors.index.json" ]]
}

run_phase() {
  local phase="$1"
  shift
  local log_file="${LOG_ROOT}/${phase}.log"
  local start_iso end_iso start_s end_s rc
  start_iso="$(date --iso-8601=seconds)"
  start_s="$(date +%s)"
  log "START ${phase}: $*"
  set +e
  "$@" 2>&1 | tee "${log_file}"
  rc="${PIPESTATUS[0]}"
  set -e
  end_s="$(date +%s)"
  end_iso="$(date --iso-8601=seconds)"
  if [[ "${rc}" -eq 0 ]]; then
    echo -e "${phase}\tok\t${start_iso}\t${end_iso}\t$((end_s-start_s))\t${log_file}" >> "${TIMINGS_TSV}"
    log "DONE ${phase} in $((end_s-start_s))s"
  else
    echo -e "${phase}\tfailed:${rc}\t${start_iso}\t${end_iso}\t$((end_s-start_s))\t${log_file}" >> "${TIMINGS_TSV}"
    log "FAILED ${phase} with rc=${rc}; see ${log_file}"
    exit "${rc}"
  fi
}

skip_phase() {
  local phase="$1"
  local reason="$2"
  local now
  now="$(date --iso-8601=seconds)"
  echo -e "${phase}\tskipped:${reason}\t${now}\t${now}\t0\t-" >> "${TIMINGS_TSV}"
  log "SKIP ${phase}: ${reason}"
}

log "Run ID: ${RUN_ID}"
log "A checkpoint: ${A_CKPT}"
log "B checkpoint: ${B_CKPT}"
log "Eval parquet: ${EVAL_PARQUET}"
log "Logs: ${LOG_ROOT}"
log "Report: ${REPORT_HTML}"

[[ -d "${A_CKPT}" ]] || { echo "missing A checkpoint: ${A_CKPT}" >&2; exit 2; }
[[ -f "${EVAL_PARQUET}" ]] || { echo "missing eval parquet: ${EVAL_PARQUET}" >&2; exit 2; }
[[ -d "${BASE_MODEL_PATH}" ]] || { echo "missing base model: ${BASE_MODEL_PATH}" >&2; exit 2; }

if [[ "${FORCE_B:-0}" == "1" || ! -s "${TRAIN_JSONL}" ]]; then
  run_phase b_rollout_train "${RL_PYTHON}" "${REPO_ROOT}/cropi/inference/generate_rollouts.py" \
    --parquet "${DATA_ROOT}/gsm8k/train_qwen.parquet" \
    --model "${BASE_MODEL_PATH}" \
    --output "${TRAIN_JSONL}" \
    --n "${N_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --max_tokens "${RL_MAX_RESPONSE_LENGTH}" \
    --max_prompt_tokens "${RL_MAX_PROMPT_LENGTH}" \
    --tp_size "${RL_TP_SIZE}" \
    --seed "${SEED}" \
    --limit "${POOL}"
else
  skip_phase b_rollout_train exists
fi

if [[ "${FORCE_B:-0}" == "1" || ! -s "${VALID_JSONL}" ]]; then
  run_phase b_rollout_valid "${RL_PYTHON}" "${REPO_ROOT}/cropi/inference/generate_rollouts.py" \
    --parquet "${DATA_ROOT}/gsm8k/valid_qwen.parquet" \
    --model "${BASE_MODEL_PATH}" \
    --output "${VALID_JSONL}" \
    --n "${N_SAMPLES_VAL}" \
    --temperature "${TEMPERATURE}" \
    --max_tokens "${RL_MAX_RESPONSE_LENGTH}" \
    --max_prompt_tokens "${RL_MAX_PROMPT_LENGTH}" \
    --tp_size "${RL_TP_SIZE}" \
    --seed "${SEED}" \
    --limit "${VPOOL}"
else
  skip_phase b_rollout_valid exists
fi

if [[ "${FORCE_B:-0}" == "1" || ! -d "${B_CKPT}" ]]; then
  run_phase b_grad bash "${REPO_ROOT}/cropi/scripts/run_cropi.sh" grad-only "${DATA_ROOT}" "${MODEL_NAME}"
  run_phase b_score_select_train bash "${REPO_ROOT}/cropi/scripts/run_cropi.sh" full "${DATA_ROOT}" "${MODEL_NAME}"
else
  skip_phase b_grad b_checkpoint_exists
  skip_phase b_score_select_train b_checkpoint_exists
fi

if ! have_hf_model "${B_CKPT}"; then
  echo "B HF checkpoint is missing or incomplete: ${B_CKPT}" >&2
  exit 3
fi

run_phase eval_a "${RL_PYTHON}" "${REPO_ROOT}/cropi/eval/eval_math.py" \
  --parquet "${EVAL_PARQUET}" \
  --model "${A_CKPT}" \
  --tag "gsm8k_armA_full_3b_step150" \
  --out "${A_RESULT}" \
  --max_tokens "${RL_MAX_RESPONSE_LENGTH}" \
  --max_prompt_tokens "${RL_MAX_PROMPT_LENGTH}" \
  --tp_size "${RL_TP_SIZE}" \
  --gpu_mem "${EVAL_GPU_MEM:-0.85}" \
  --overwrite

run_phase eval_b "${RL_PYTHON}" "${REPO_ROOT}/cropi/eval/eval_math.py" \
  --parquet "${EVAL_PARQUET}" \
  --model "${B_CKPT}" \
  --tag "gsm8k_armB_cropi10_3b_step${RL_TOTAL_TRAINING_STEPS}" \
  --out "${B_RESULT}" \
  --max_tokens "${RL_MAX_RESPONSE_LENGTH}" \
  --max_prompt_tokens "${RL_MAX_PROMPT_LENGTH}" \
  --tp_size "${RL_TP_SIZE}" \
  --gpu_mem "${EVAL_GPU_MEM:-0.85}" \
  --overwrite

run_phase make_report python3 "${REPO_ROOT}/scripts/make_gsm8k_ab_report.py" \
  --results-dir "${RESULTS_DIR}" \
  --timings "${TIMINGS_TSV}" \
  --out "${REPORT_HTML}" \
  --a-result "${A_RESULT}" \
  --b-result "${B_RESULT}" \
  --a-ckpt-root "${A_CKPT_ROOT}" \
  --b-ckpt-root "${B_CKPT_ROOT}" \
  --b-data-dir "${DATA_ROOT}/gsm8k/${MODEL_NAME}" \
  --eval-parquet "${EVAL_PARQUET}" \
  --run-id "${RUN_ID}" \
  --repo "${REPO_ROOT}"

log "A result: ${A_RESULT}"
log "B result: ${B_RESULT}"
log "HTML report: ${REPORT_HTML}"
log "timings: ${TIMINGS_TSV}"
