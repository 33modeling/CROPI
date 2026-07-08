#!/bin/bash
# Smoke test: RL-only round on the already-selected gsm8k parquet (2x4090).
export GROUP_VOLUME=/data1
export VLLM_USE_V1=1
export BASE_MODEL_PATH="/data1/minsoo3.kim/weasel/models/Qwen3.5-9B"
export TRAIN_DATA_NAMES="gsm8k" VALID_DATA_NAMES="gsm8k" RL_VAL_DATA_NAMES="gsm8k"
export N_SAMPLES=4 N_SAMPLES_VAL=4 PROMPT_TYPE="qwen25-math-cot" SEED=0 TEMPERATURE=0.5
export INFER_NOTE="${PROMPT_TYPE}_-1_seed${SEED}_t${TEMPERATURE}_n${N_SAMPLES}_s0_e-1"
export NUM_PARALLEL=1 RL_TRAIN_BATCH_SIZE=16 RL_PPO_MINI_BATCH_SIZE=16
export RL_PPO_MICRO_BATCH_SIZE_PER_GPU=2 RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2
export RL_FSDP_TRANSFORMER_LAYER_CLS=Qwen3_5DecoderLayer
source scripts/setup_env.sh >/dev/null 2>&1
echo "RL_PYTHON=$RL_PYTHON RL_NUM_GPUS=$RL_NUM_GPUS RL_TP_SIZE=$RL_TP_SIZE"
exec bash cropi/scripts/run_cropi.sh rl-only "$DATA_ROOT" "Qwen3.5-9B_curriculum"
