#!/bin/bash
# A100-mirror smoke test: RL-only round with Qwen2.5-7B-Instruct on 2x4090.
# Mirrors the A100 plan (Qwen2.5-7B, vllm 0.8.5, transformers<5). Reuses the
# existing gsm8k selected parquet (its content is model-agnostic gsm8k prompts).
# NOTE: RL_FSDP_TRANSFORMER_LAYER_CLS is intentionally UNSET — Qwen2's
# _no_split_modules (Qwen2DecoderLayer only) is valid, so no wrap override needed.
export GROUP_VOLUME=/data1
export VLLM_USE_V1=1
export BASE_MODEL_PATH="/data1/minsoo3.kim/weasel/models/Qwen2.5-7B-Instruct"
export TRAIN_DATA_NAMES="gsm8k" VALID_DATA_NAMES="gsm8k" RL_VAL_DATA_NAMES="gsm8k"
export N_SAMPLES=4 N_SAMPLES_VAL=4 PROMPT_TYPE="qwen25-math-cot" SEED=0 TEMPERATURE=0.5
export INFER_NOTE="${PROMPT_TYPE}_-1_seed${SEED}_t${TEMPERATURE}_n${N_SAMPLES}_s0_e-1"
export NUM_PARALLEL=1 RL_TRAIN_BATCH_SIZE=16 RL_PPO_MINI_BATCH_SIZE=16
export RL_PPO_MICRO_BATCH_SIZE_PER_GPU=2 RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2
# --- 24GB (2x4090) fit: CPU-offload actor + shard vLLM rollout across both GPUs ---
export RL_PARAM_OFFLOAD=True RL_OPTIMIZER_OFFLOAD=True
export RL_TP_SIZE=2 RL_GPU_MEMORY_UTILIZATION=0.5
export RL_MAX_RESPONSE_LENGTH=1024 RL_MAX_PROMPT_LENGTH=1024 RL_N_SAMPLES=2
source scripts/setup_env.sh >/dev/null 2>&1
echo "RL_PYTHON=$RL_PYTHON RL_NUM_GPUS=$RL_NUM_GPUS RL_TP_SIZE=$RL_TP_SIZE BASE_MODEL_PATH=$BASE_MODEL_PATH"
exec bash cropi/scripts/run_cropi.sh rl-only "$DATA_ROOT" "Qwen3.5-9B_curriculum"
