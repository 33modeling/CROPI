#!/bin/bash
export GROUP_VOLUME=/data1
export VLLM_USE_V1=1
export BASE_MODEL_PATH="/data1/minsoo3.kim/weasel/models/Qwen3.5-9B"
export TRAIN_DATA_NAMES="gsm8k"
export VALID_DATA_NAMES="gsm8k"
export RL_VAL_DATA_NAMES="gsm8k"
export N_SAMPLES=4
export N_SAMPLES_VAL=4
export PROMPT_TYPE="qwen25-math-cot"
export SEED=0
export TEMPERATURE=0.5
export INFER_NOTE="${PROMPT_TYPE}_-1_seed${SEED}_t${TEMPERATURE}_n${N_SAMPLES}_s0_e-1"
export NUM_PARALLEL=1

# 데이터가 29개뿐이므로, 기존 128이던 배치 사이즈를 데이터 개수보다 작은 16으로 팍 줄입니다.
export RL_TRAIN_BATCH_SIZE=16
export RL_PPO_MINI_BATCH_SIZE=16

# 마이크로 배치도 전체 배치보다 작아야 하므로 2로 줄입니다. (gpu 당)
export RL_PPO_MICRO_BATCH_SIZE_PER_GPU=2
export RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2

source scripts/setup_env.sh

bash cropi/scripts/run_cropi.sh full "$DATA_ROOT" "Qwen3.5-9B_curriculum"
