#!/bin/bash
# Arm B: CROPI 10% selection (SINGLE round) on Qwen2.5-3B, gsm8k.
# Scope reduced from 3-round/full-7473 because get-grad ~= 18s/example on 3B/4090
# (full 7473 x 3 rounds would be ~55h). Candidate pool = 2000 train, valid ref = 256.
# Pipeline: rollout(base 3B) -> grad-only -> full(NUM_RL_ROUNDS=1 => score+select+train).
set -uo pipefail
cd /home/minsoo3.kim/dev/CROPI
export GROUP_VOLUME=/data1 VLLM_USE_V1=1
export BASE_MODEL_PATH="/data1/minsoo3.kim/weasel/models/Qwen2.5-3B-Instruct"
export TRAIN_DATA_NAMES="gsm8k" VALID_DATA_NAMES="gsm8k" RL_VAL_DATA_NAMES="gsm8k"
export PROMPT_TYPE="qwen25-math-cot" SEED=0 TEMPERATURE=0.5
export N_SAMPLES=4 N_SAMPLES_VAL=4
export RL_MAX_RESPONSE_LENGTH=1024 RL_MAX_PROMPT_LENGTH=1024 RL_TP_SIZE=2 RL_GPU_MEMORY_UTILIZATION=0.5
export NUM_PARALLEL=2                    # grad across both GPUs
export SELECT_RATIO=0.1
export INFER_NOTE="${PROMPT_TYPE}_-1_seed${SEED}_t${TEMPERATURE}_n${N_SAMPLES}_s0_e-1"
export VALID_INFER_NOTE="${PROMPT_TYPE}_-1_seed${SEED}_t${TEMPERATURE}_n${N_SAMPLES_VAL}_s0_e-1"
# RL training params for the single round (matched to Arm A)
export RL_TRAIN_BATCH_SIZE=32 RL_PPO_MINI_BATCH_SIZE=32
export RL_PPO_MICRO_BATCH_SIZE_PER_GPU=2 RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2
export RL_TOTAL_TRAINING_STEPS=150 RL_SAVE_FREQ=150 RL_TEST_FREQ=50
export NUM_RL_ROUNDS=1
export CUSTOM_REWARD_PATH="$(pwd)/cropi/rewards/gsm8k_math.py" CUSTOM_REWARD_NAME=compute_score
export PYTHONPATH=/home/minsoo3.kim/dev/CROPI
# Use the cropi venv's console scripts directly (avoid `uv run` network/TLS sync).
export CROPI_RUN=""
export PATH=/data1/minsoo3.kim/cropi/venvs/cropi/bin:$PATH
source scripts/setup_env.sh >/dev/null 2>&1

MODEL_NAME="Qwen2.5-3B_curriculum"
RP=/data1/minsoo3.kim/cropi/venvs/verl/bin/python
DIR="$DATA_ROOT/gsm8k/$MODEL_NAME"; mkdir -p "$DIR"
TRAIN_JSONL="$DIR/train_${INFER_NOTE}.jsonl"
VALID_JSONL="$DIR/valid_${VALID_INFER_NOTE}.jsonl"
POOL=${POOL:-2000}; VPOOL=${VPOOL:-256}

echo "########## [1/3] ROLLOUTS (base 3B: train limit=$POOL, valid limit=$VPOOL) ##########"
if [[ ! -s "$TRAIN_JSONL" ]]; then
  "$RP" cropi/inference/generate_rollouts.py --parquet "$DATA_ROOT/gsm8k/train_qwen.parquet" \
    --model "$BASE_MODEL_PATH" --output "$TRAIN_JSONL" --n "$N_SAMPLES" --temperature "$TEMPERATURE" \
    --max_tokens "$RL_MAX_RESPONSE_LENGTH" --max_prompt_tokens "$RL_MAX_PROMPT_LENGTH" \
    --tp_size "$RL_TP_SIZE" --seed "$SEED" --limit "$POOL" || { echo "ROLLOUT train FAILED"; exit 1; }
fi
if [[ ! -s "$VALID_JSONL" ]]; then
  "$RP" cropi/inference/generate_rollouts.py --parquet "$DATA_ROOT/gsm8k/valid_qwen.parquet" \
    --model "$BASE_MODEL_PATH" --output "$VALID_JSONL" --n "$N_SAMPLES_VAL" --temperature "$TEMPERATURE" \
    --max_tokens "$RL_MAX_RESPONSE_LENGTH" --max_prompt_tokens "$RL_MAX_PROMPT_LENGTH" \
    --tp_size "$RL_TP_SIZE" --seed "$SEED" --limit "$VPOOL" || { echo "ROLLOUT valid FAILED"; exit 1; }
fi
echo "rollouts: train=$(wc -l <"$TRAIN_JSONL") valid=$(wc -l <"$VALID_JSONL")"

echo "########## [2/3] GRAD-ONLY (base 3B gradients, NUM_PARALLEL=$NUM_PARALLEL) ##########"
bash cropi/scripts/run_cropi.sh grad-only "$DATA_ROOT" "$MODEL_NAME" || { echo "GRAD FAILED"; exit 1; }

echo "########## [3/3] FULL single round: score+select(10%)+train 150 steps ##########"
bash cropi/scripts/run_cropi.sh full "$DATA_ROOT" "$MODEL_NAME" || { echo "FULL FAILED"; exit 1; }
echo "########## ARM B DONE ##########"
