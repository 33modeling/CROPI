#!/bin/bash
export GROUP_VOLUME=/data1
export VLLM_USE_V1=1
export BASE_MODEL_PATH="/data1/minsoo3.kim/weasel/models/Qwen3.5-9B"

# math 제거
export TRAIN_DATA_NAMES="gsm8k"
export VALID_DATA_NAMES="gsm8k"
export RL_VAL_DATA_NAMES="gsm8k"

# n4 설정 덮어쓰기 (기존 데이터와 매칭)
export N_SAMPLES=4
export N_SAMPLES_VAL=4
export PROMPT_TYPE="qwen25-math-cot"
export SEED=0
export TEMPERATURE=0.5
export INFER_NOTE="${PROMPT_TYPE}_-1_seed${SEED}_t${TEMPERATURE}_n${N_SAMPLES}_s0_e-1"

source scripts/setup_env.sh
bash cropi/scripts/run_cropi.sh full "$DATA_ROOT" "gsm8k" "Qwen3.5-9B_curriculum"
