#!/bin/bash
# Arm A (baseline): full-data GRPO on ALL 7473 gsm8k examples, Qwen2.5-3B, 150 steps.
export GROUP_VOLUME=/data1
export VLLM_USE_V1=1
export BASE_MODEL_PATH="/data1/minsoo3.kim/weasel/models/Qwen2.5-3B-Instruct"
export TRAIN_DATA_NAMES="gsm8k" VALID_DATA_NAMES="gsm8k" RL_VAL_DATA_NAMES="gsm8k"
export PROMPT_TYPE="qwen25-math-cot" SEED=0 TEMPERATURE=0.5
# full-data run: 150 steps, batch 32
export RL_TRAIN_BATCH_SIZE=32 RL_PPO_MINI_BATCH_SIZE=32
export RL_PPO_MICRO_BATCH_SIZE_PER_GPU=2 RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2
export RL_TP_SIZE=2 RL_GPU_MEMORY_UTILIZATION=0.5
export RL_MAX_RESPONSE_LENGTH=1024 RL_MAX_PROMPT_LENGTH=1024 RL_N_SAMPLES=4
export RL_TOTAL_TRAINING_STEPS=150 RL_SAVE_FREQ=50 RL_TEST_FREQ=50
export BASELINE_EXP_NAME="gsm8k_full_3b"
export CUSTOM_REWARD_PATH="$(pwd)/cropi/rewards/gsm8k_math.py" CUSTOM_REWARD_NAME=compute_score
source scripts/setup_env.sh >/dev/null 2>&1
echo "ArmA base=$BASE_MODEL_PATH steps=$RL_TOTAL_TRAINING_STEPS batch=$RL_TRAIN_BATCH_SIZE ckpt=$CKPT_ROOT/cropi_rl/$BASELINE_EXP_NAME"
exec bash cropi/scripts/run_cropi.sh baseline-full "$DATA_ROOT" "Qwen2.5-3B-Instruct_armA"
