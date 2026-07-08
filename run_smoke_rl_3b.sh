#!/bin/bash
# Small-model full run on 2x4090: Qwen2.5-3B-Instruct, gsm8k, 10 GRPO steps to completion.
# Same Qwen2ForCausalLM code path as the A100 Qwen2.5-7B target; 3B fits 24GB with room,
# so NO CPU offload (closer to the A100 default config). Reuses the existing gsm8k
# selected parquet (model-agnostic gsm8k prompts).
export GROUP_VOLUME=/data1
export VLLM_USE_V1=1
export BASE_MODEL_PATH="/data1/minsoo3.kim/weasel/models/Qwen2.5-3B-Instruct"
export TRAIN_DATA_NAMES="gsm8k" VALID_DATA_NAMES="gsm8k" RL_VAL_DATA_NAMES="gsm8k"
export N_SAMPLES=4 N_SAMPLES_VAL=4 PROMPT_TYPE="qwen25-math-cot" SEED=0 TEMPERATURE=0.5
export INFER_NOTE="${PROMPT_TYPE}_-1_seed${SEED}_t${TEMPERATURE}_n${N_SAMPLES}_s0_e-1"
export NUM_PARALLEL=1 RL_TRAIN_BATCH_SIZE=16 RL_PPO_MINI_BATCH_SIZE=16
export RL_PPO_MICRO_BATCH_SIZE_PER_GPU=2 RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2
# rollout across both GPUs; 3B leaves plenty of KV headroom at 0.5
export RL_TP_SIZE=2 RL_GPU_MEMORY_UTILIZATION=0.5
export RL_MAX_RESPONSE_LENGTH=1024 RL_MAX_PROMPT_LENGTH=1024 RL_N_SAMPLES=4
# 10 steps, save + validate at the end -> produces a checkpoint and a val score
export RL_TOTAL_TRAINING_STEPS=10 RL_SAVE_FREQ=10 RL_TEST_FREQ=5
export CUSTOM_REWARD_PATH="$(pwd)/cropi/rewards/gsm8k_math.py" CUSTOM_REWARD_NAME=compute_score
source scripts/setup_env.sh >/dev/null 2>&1
echo "RL_PYTHON=$RL_PYTHON RL_NUM_GPUS=$RL_NUM_GPUS RL_TP_SIZE=$RL_TP_SIZE BASE=$BASE_MODEL_PATH"
exec bash cropi/scripts/run_cropi.sh rl-only "$DATA_ROOT" "Qwen3.5-9B_curriculum"
