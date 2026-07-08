#!/bin/bash
export GROUP_VOLUME=/data1
export VLLM_USE_V1=1
export BASE_MODEL_PATH="/data1/minsoo3.kim/weasel/models/Qwen3.5-9B"

# math 데이터셋 의존성 제거를 위해 환경변수 오버라이드
export TRAIN_DATA_NAMES="gsm8k"
export VALID_DATA_NAMES="gsm8k"
export RL_VAL_DATA_NAMES="gsm8k"

source scripts/setup_env.sh

# Qwen3.5-9B_curriculum 데이터를 사용하도록 인자 지정
bash cropi/scripts/run_cropi.sh full "$DATA_ROOT" "gsm8k" "Qwen3.5-9B_curriculum"
