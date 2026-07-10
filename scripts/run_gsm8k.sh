#!/usr/bin/env bash
# End-to-end gsm8k experiment: full-data GRPO (Arm A) vs CROPI 10% 3-round
# curriculum (Arm B), matched on TOTAL training steps, evaluated on gsm8k test.
#
#   source scripts/setup_env_a100.sh 4        # or 8
#   cropi_activate
#   bash scripts/run_gsm8k.sh                  # run everything (idempotent)
#   DRY_RUN=1 bash scripts/run_gsm8k.sh        # print the command chain only
#   PHASES=prep,baseline bash scripts/run_gsm8k.sh   # run a subset
#
# Phases: preflight, prep, baseline, rollout, grad, cropi, eval, compare
# Every phase is skip-if-done, so re-running resumes where it stopped.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

: "${DATA_ROOT:?source scripts/setup_env_a100.sh first}"
: "${BASE_MODEL_PATH:?source scripts/setup_env_a100.sh first}"
: "${RL_PYTHON:?source scripts/setup_env_a100.sh first}"

DATASET="gsm8k"
export TRAIN_DATA_NAMES="${DATASET}"
export VALID_DATA_NAMES="${DATASET}"
export RL_VAL_DATA_NAMES="${DATASET}"
export RL_PROJECT_NAME="${DATASET}"
MODEL_NAME="${INITIAL_MODEL_NAME:-Qwen2.5-7B-Instruct_curriculum}"

INFER_NOTE="${PROMPT_TYPE}_-1_seed${SEED}_t${TEMPERATURE}_n${N_SAMPLES}_s0_e-1"
VALID_INFER_NOTE="${PROMPT_TYPE}_-1_seed${SEED}_t${TEMPERATURE}_n${N_SAMPLES_VAL}_s0_e-1"

# Matched-compute step budget: baseline uses TOTAL, CROPI splits it across rounds.
TOTAL_STEPS="${RL_TOTAL_TRAINING_STEPS:-60}"
CROPI_PER_ROUND=$(( TOTAL_STEPS / NUM_RL_ROUNDS ))
[[ ${CROPI_PER_ROUND} -ge 1 ]] || CROPI_PER_ROUND=1

DRY_RUN="${DRY_RUN:-0}"
PHASES="${PHASES:-preflight,prep,baseline,rollout,grad,cropi,eval,compare}"
RESULTS_DIR="${RESULTS_DIR:-${CROPI_WORK}/results}"
mkdir -p "${RESULTS_DIR}"

log(){ echo -e "\n\033[1;36m[run_gsm8k] $*\033[0m"; }
want(){ [[ ",${PHASES}," == *",$1,"* ]]; }
run(){ if [[ "${DRY_RUN}" == "1" ]]; then echo "[DRY] $*"; else eval "$@"; fi; }

BASELINE_CKPT="${CKPT_ROOT}/${RL_PROJECT_NAME}/full/global_step_${TOTAL_STEPS}/actor/huggingface"
CROPI_LAST=$(( NUM_RL_ROUNDS - 1 ))
CROPI_CKPT="${CKPT_ROOT}/${RL_PROJECT_NAME}/iter${CROPI_LAST}/global_step_${CROPI_PER_ROUND}/actor/huggingface"

# ---------------------------------------------------------------- preflight ---
if want preflight; then
  log "Phase 0: preflight"
  bash "${SCRIPT_DIR}/preflight.sh" || echo "[run_gsm8k] preflight reported issues — review above."
fi

# ------------------------------------------------------------------- prep -----
if want prep; then
  log "Phase 1: preprocess gsm8k -> parquet"
  run "python '${REPO_ROOT}/cropi/data_prep/prep_gsm8k.py' \
        --raw_dir '${IT_DATASETS}/${DATASET}' --out_dir '${DATA_ROOT}/${DATASET}' --seed '${SEED}'"
fi

# --------------------------------------------------- Arm A: full baseline -----
if want baseline; then
  log "Phase 2 (Arm A): full-data GRPO, ${TOTAL_STEPS} steps"
  if [[ -d "${BASELINE_CKPT}" && "${DRY_RUN}" != "1" ]]; then
    echo "[skip] baseline checkpoint exists: ${BASELINE_CKPT}"
  else
    RL_TOTAL_TRAINING_STEPS="${TOTAL_STEPS}" RL_SAVE_FREQ="${TOTAL_STEPS}" BASELINE_EXP_NAME="full" \
      run "bash '${SCRIPT_DIR%/}/../cropi/scripts/run_cropi.sh' baseline-full '${DATA_ROOT}' '${MODEL_NAME}'"
  fi
fi

# ------------------------------------------- Arm B step 1: rollouts (base) ----
if want rollout; then
  log "Phase 3 (Arm B): rollouts from base model (train + valid)"
  train_jsonl="${DATA_ROOT}/${DATASET}/${MODEL_NAME}/train_${INFER_NOTE}.jsonl"
  valid_jsonl="${DATA_ROOT}/${DATASET}/${MODEL_NAME}/valid_${VALID_INFER_NOTE}.jsonl"
  run "'${RL_PYTHON}' '${REPO_ROOT}/cropi/inference/generate_rollouts.py' \
        --parquet '${DATA_ROOT}/${DATASET}/train_qwen.parquet' --model '${BASE_MODEL_PATH}' \
        --output '${train_jsonl}' --n '${N_SAMPLES}' --temperature '${TEMPERATURE}' \
        --max_tokens '${RL_MAX_RESPONSE_LENGTH}' --max_prompt_tokens '${RL_MAX_PROMPT_LENGTH}' \
        --tp_size '${RL_TP_SIZE}' --seed '${SEED}'"
  run "'${RL_PYTHON}' '${REPO_ROOT}/cropi/inference/generate_rollouts.py' \
        --parquet '${DATA_ROOT}/${DATASET}/valid_qwen.parquet' --model '${BASE_MODEL_PATH}' \
        --output '${valid_jsonl}' --n '${N_SAMPLES_VAL}' --temperature '${TEMPERATURE}' \
        --max_tokens '${RL_MAX_RESPONSE_LENGTH}' --max_prompt_tokens '${RL_MAX_PROMPT_LENGTH}' \
        --tp_size '${RL_TP_SIZE}' --seed '${SEED}'"
fi

# --------------------------------------- Arm B step 2: round-0 gradients ------
if want grad; then
  log "Phase 4 (Arm B): round-0 projected gradients (uv/cropi env)"
  run "bash '${REPO_ROOT}/cropi/scripts/run_cropi.sh' grad-only '${DATA_ROOT}' '${MODEL_NAME}'"
fi

# ------------------------------ Arm B step 3: 3-round CROPI select -> RL -------
if want cropi; then
  log "Phase 5 (Arm B): CROPI ${NUM_RL_ROUNDS}-round curriculum (${CROPI_PER_ROUND} steps/round = ${TOTAL_STEPS} total)"
  RL_TOTAL_TRAINING_STEPS="${CROPI_PER_ROUND}" RL_SAVE_FREQ="${CROPI_PER_ROUND}" \
    run "bash '${REPO_ROOT}/cropi/scripts/run_cropi.sh' full '${DATA_ROOT}' '${MODEL_NAME}'"
fi

# ------------------------------------------------------------------- eval -----
if want eval; then
  log "Phase 6: evaluate both arms on gsm8k test"
  test_parquet="${DATA_ROOT}/${DATASET}/test_qwen.parquet"
  run "'${RL_PYTHON}' '${REPO_ROOT}/cropi/eval/eval_math.py' --parquet '${test_parquet}' \
        --model '${BASELINE_CKPT}' --tag '${DATASET}_full' --out '${RESULTS_DIR}/${DATASET}_full.json'"
  run "'${RL_PYTHON}' '${REPO_ROOT}/cropi/eval/eval_math.py' --parquet '${test_parquet}' \
        --model '${CROPI_CKPT}' --tag '${DATASET}_cropi10' --out '${RESULTS_DIR}/${DATASET}_cropi10.json'"
fi

# ---------------------------------------------------------------- compare -----
if want compare; then
  log "Phase 7: comparison report"
  run "'${RL_PYTHON}' '${REPO_ROOT}/cropi/eval/make_report.py' \
        --results-dir '${RESULTS_DIR}' --dataset '${DATASET}' \
        --steps '${TOTAL_STEPS}' --rounds '${NUM_RL_ROUNDS}' --select-ratio 10 \
        --out '${RESULTS_DIR}/${DATASET}_report.md'"
fi

log "done. results in ${RESULTS_DIR}"
