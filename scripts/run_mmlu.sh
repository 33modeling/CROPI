#!/usr/bin/env bash
# End-to-end MMLU experiment: full-data GRPO (Arm A) vs CROPI 10% 3-round
# curriculum (Arm B), matched on TOTAL training steps, evaluated on MMLU test.
# MMLU is 4-way multiple choice -> letter-match reward (cropi/rewards/choice.py),
# used for rollouts, RL (verl custom_reward_function), and eval.
#
#   source scripts/setup_env_a100.sh 2      # or 4/8
#   cropi_activate
#   bash scripts/run_mmlu.sh                 # DRY_RUN=1 to preview; PHASES=... to subset
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

: "${DATA_ROOT:?source scripts/setup_env_a100.sh first}"
: "${BASE_MODEL_PATH:?source scripts/setup_env_a100.sh first}"
: "${RL_PYTHON:?source scripts/setup_env_a100.sh first}"

DATASET="mmlu"
export TRAIN_DATA_NAMES="${DATASET}"
export VALID_DATA_NAMES="${DATASET}"
export RL_VAL_DATA_NAMES="${DATASET}"
export RL_PROJECT_NAME="${DATASET}"
MODEL_NAME="${INITIAL_MODEL_NAME:-Qwen2.5-7B-Instruct_curriculum}"

# MMLU = choice task: letter-match reward for verl RL (both arms).
export CUSTOM_REWARD_PATH="${REPO_ROOT}/cropi/rewards/choice.py"
export CUSTOM_REWARD_NAME="compute_score"

INFER_NOTE="${PROMPT_TYPE}_-1_seed${SEED}_t${TEMPERATURE}_n${N_SAMPLES}_s0_e-1"
VALID_INFER_NOTE="${PROMPT_TYPE}_-1_seed${SEED}_t${TEMPERATURE}_n${N_SAMPLES_VAL}_s0_e-1"

TOTAL_STEPS="${RL_TOTAL_TRAINING_STEPS:-60}"
CROPI_PER_ROUND=$(( TOTAL_STEPS / NUM_RL_ROUNDS ))
[[ ${CROPI_PER_ROUND} -ge 1 ]] || CROPI_PER_ROUND=1

DRY_RUN="${DRY_RUN:-0}"
PHASES="${PHASES:-preflight,prep,baseline,rollout,grad,cropi,eval,compare}"
RESULTS_DIR="${RESULTS_DIR:-${CROPI_WORK}/results}"
mkdir -p "${RESULTS_DIR}"

log(){ echo -e "\n\033[1;35m[run_mmlu] $*\033[0m"; }
want(){ [[ ",${PHASES}," == *",$1,"* ]]; }
run(){ if [[ "${DRY_RUN}" == "1" ]]; then echo "[DRY] $*"; else eval "$@"; fi; }

BASELINE_CKPT="${CKPT_ROOT}/${RL_PROJECT_NAME}/full/global_step_${TOTAL_STEPS}/actor/huggingface"
CROPI_LAST=$(( NUM_RL_ROUNDS - 1 ))
CROPI_CKPT="${CKPT_ROOT}/${RL_PROJECT_NAME}/iter${CROPI_LAST}/global_step_${CROPI_PER_ROUND}/actor/huggingface"

if want preflight; then
  log "Phase 0: preflight"
  bash "${SCRIPT_DIR}/preflight.sh" || echo "[run_mmlu] preflight reported issues — review above."
fi

if want prep; then
  log "Phase 1: preprocess mmlu -> parquet (choice)"
  run "python '${REPO_ROOT}/cropi/data_prep/prep_mmlu.py' \
        --raw_dir '${IT_DATASETS}/${DATASET}' --out_dir '${DATA_ROOT}/${DATASET}' --seed '${SEED}'"
fi

if want baseline; then
  log "Phase 2 (Arm A): full-data GRPO, ${TOTAL_STEPS} steps (letter reward)"
  if [[ -d "${BASELINE_CKPT}" && "${DRY_RUN}" != "1" ]]; then
    echo "[skip] baseline checkpoint exists: ${BASELINE_CKPT}"
  else
    RL_TOTAL_TRAINING_STEPS="${TOTAL_STEPS}" RL_SAVE_FREQ="${TOTAL_STEPS}" BASELINE_EXP_NAME="full" \
      run "bash '${REPO_ROOT}/cropi/scripts/run_cropi.sh' baseline-full '${DATA_ROOT}' '${MODEL_NAME}'"
  fi
fi

if want rollout; then
  log "Phase 3 (Arm B): rollouts from base model (train + valid, --reward choice)"
  train_jsonl="${DATA_ROOT}/${DATASET}/${MODEL_NAME}/train_${INFER_NOTE}.jsonl"
  valid_jsonl="${DATA_ROOT}/${DATASET}/${MODEL_NAME}/valid_${VALID_INFER_NOTE}.jsonl"
  run "'${RL_PYTHON}' '${REPO_ROOT}/cropi/inference/generate_rollouts.py' \
        --parquet '${DATA_ROOT}/${DATASET}/train_qwen.parquet' --model '${BASE_MODEL_PATH}' \
        --output '${train_jsonl}' --n '${N_SAMPLES}' --temperature '${TEMPERATURE}' --reward choice \
        --max_tokens '${RL_MAX_RESPONSE_LENGTH}' --max_prompt_tokens '${RL_MAX_PROMPT_LENGTH}' \
        --tp_size '${RL_TP_SIZE}' --seed '${SEED}'"
  run "'${RL_PYTHON}' '${REPO_ROOT}/cropi/inference/generate_rollouts.py' \
        --parquet '${DATA_ROOT}/${DATASET}/valid_qwen.parquet' --model '${BASE_MODEL_PATH}' \
        --output '${valid_jsonl}' --n '${N_SAMPLES_VAL}' --temperature '${TEMPERATURE}' --reward choice \
        --max_tokens '${RL_MAX_RESPONSE_LENGTH}' --max_prompt_tokens '${RL_MAX_PROMPT_LENGTH}' \
        --tp_size '${RL_TP_SIZE}' --seed '${SEED}'"
fi

if want grad; then
  log "Phase 4 (Arm B): round-0 projected gradients"
  run "bash '${REPO_ROOT}/cropi/scripts/run_cropi.sh' grad-only '${DATA_ROOT}' '${MODEL_NAME}'"
fi

if want cropi; then
  log "Phase 5 (Arm B): CROPI ${NUM_RL_ROUNDS}-round curriculum (${CROPI_PER_ROUND} steps/round)"
  RL_TOTAL_TRAINING_STEPS="${CROPI_PER_ROUND}" RL_SAVE_FREQ="${CROPI_PER_ROUND}" \
    run "bash '${REPO_ROOT}/cropi/scripts/run_cropi.sh' full '${DATA_ROOT}' '${MODEL_NAME}'"
fi

if want eval; then
  log "Phase 6: evaluate both arms on mmlu test (--reward choice)"
  test_parquet="${DATA_ROOT}/${DATASET}/test_qwen.parquet"
  run "'${RL_PYTHON}' '${REPO_ROOT}/cropi/eval/eval_math.py' --parquet '${test_parquet}' --reward choice \
        --model '${BASELINE_CKPT}' --tag '${DATASET}_full' --out '${RESULTS_DIR}/${DATASET}_full.json'"
  run "'${RL_PYTHON}' '${REPO_ROOT}/cropi/eval/eval_math.py' --parquet '${test_parquet}' --reward choice \
        --model '${CROPI_CKPT}' --tag '${DATASET}_cropi10' --out '${RESULTS_DIR}/${DATASET}_cropi10.json'"
fi

if want compare; then
  log "Phase 7: comparison report"
  run "'${RL_PYTHON}' '${REPO_ROOT}/cropi/eval/make_report.py' \
        --results-dir '${RESULTS_DIR}' --dataset '${DATASET}' \
        --steps '${TOTAL_STEPS}' --rounds '${NUM_RL_ROUNDS}' --select-ratio 10 \
        --out '${RESULTS_DIR}/${DATASET}_report.md'"
fi

log "done. results in ${RESULTS_DIR}"
