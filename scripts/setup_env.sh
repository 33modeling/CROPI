#!/usr/bin/env bash
# Source this file once per shell before running any CROPI stage:
#   source scripts/setup_env.sh
#
# Mirrors the conventions of our weasel/tads scripts:
# - Exports every path the install / run scripts reference.
# - Sends ALL heavy artefacts (uv venvs, models, data, HF cache, checkpoints)
#   to a single workspace ($CROPI_WORK) so the small home/user disk stays clean.
# - **Warns** (does not error) about any path that doesn't exist yet, and prints
#   the exact command to create it. Never aborts your shell.
# - Provides `cropi_activate` to enter the uv-managed CROPI env.
# - Ships defaults tuned for a 2x RTX 4090 (24GB) VM, not the paper's 8xA100.
#
# Two environments (their deps genuinely conflict — keep them separate):
#   cropi : uv venv, torch 2.4.0 cu124 + traker/fast_jl -> scoring & selection
#           (cropi-get-grad / cropi-compute-inf-score / cropi-select)
#   verl  : separate interpreter with `verl` installed -> RL (GRPO) + vLLM rollout
#           (pointed to by RL_PYTHON; heavy, version-coupled with vllm/torch)

# -----------------------------------------------------------------------------
# Workspace (override BEFORE sourcing if your mounts differ)
# -----------------------------------------------------------------------------
# On the managed cluster this is /group-volume; on a plain cloud VM point it at
# whatever large disk you have (e.g. export GROUP_VOLUME=/mnt/data before sourcing).
export GROUP_VOLUME="${GROUP_VOLUME:-/group-volume}"
export CROPI_USER="${CROPI_USER:-${USER:-$(whoami)}}"

# Repo root (this checkout). Works when sourced via BASH_SOURCE.
export CROPI_REPO="${CROPI_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

# Single workspace root. Everything large hangs off this.
export CROPI_WORK="${CROPI_WORK:-$GROUP_VOLUME/$CROPI_USER/cropi}"

# Make sure a locally-installed uv is reachable (its installer drops it here).
case ":$PATH:" in *":$HOME/.local/bin:"*) : ;; *) export PATH="$HOME/.local/bin:$PATH" ;; esac

# Best-effort visible GPU count for VM defaults. User-provided values still win.
if command -v nvidia-smi >/dev/null 2>&1; then
  _cropi_detected_gpus="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')"
else
  _cropi_detected_gpus=""
fi
case "$_cropi_detected_gpus" in
  ""|0) _cropi_detected_gpus=2 ;;
esac
export CROPI_DETECTED_GPUS="${CROPI_DETECTED_GPUS:-$_cropi_detected_gpus}"

# -----------------------------------------------------------------------------
# uv-managed environments (on the workspace disk, not in the repo)
# -----------------------------------------------------------------------------
export CROPI_VENV="${CROPI_VENV:-$CROPI_WORK/venvs/cropi}"
export VERL_VENV="${VERL_VENV:-$CROPI_WORK/venvs/verl}"
# `uv run` (used inside cropi/scripts/*) resolves this instead of a repo-local .venv.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$CROPI_VENV}"
# The interpreter that has `verl` installed (RL modes only).
export RL_PYTHON="${RL_PYTHON:-$VERL_VENV/bin/python}"
export CROPI_PY="${CROPI_PY:-3.11}"          # python version for the cropi uv venv

# -----------------------------------------------------------------------------
# Data + models + outputs (all on the workspace disk)
# -----------------------------------------------------------------------------
export DATA_ROOT="${DATA_ROOT:-$CROPI_WORK/data}"          # data/<dataset>/<model>/... layout
export MODELS_DIR="${MODELS_DIR:-$CROPI_WORK/models}"
# First-round HF checkpoint for RL. Paper uses Qwen2.5-1.5B-Instruct.
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MODELS_DIR/Qwen2.5-1.5B-Instruct}"
export HFID_BASE_MODEL="${HFID_BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"   # for a manual download
export CKPT_ROOT="${CKPT_ROOT:-$CROPI_WORK/checkpoints}"   # verl actor checkpoints + HF exports

# External rollout generator (Qwen2.5-Math eval pipeline, vLLM). Only needed if you
# regenerate rollout/gradient assets yourself via cropi/inference/infer_*.sh.
export MATH_EVAL_ENTRYPOINT="${MATH_EVAL_ENTRYPOINT:-}"    # path to math_eval_save_logprob.py

# -----------------------------------------------------------------------------
# HF cache redirect (protect the small home disk)
# -----------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-$CROPI_WORK/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CROPI_WORK/cache/uv}"

# -----------------------------------------------------------------------------
# VM pipeline defaults. Override per-invocation freely.
# cropi/scripts/run_cropi.sh reads all of these via ${VAR:-default}, so exporting
# them here avoids editing the repo's scripts.
# -----------------------------------------------------------------------------
export RL_NUM_GPUS="${RL_NUM_GPUS:-$CROPI_DETECTED_GPUS}"        # FSDP world size for RL (paper: 8)
export RL_TP_SIZE="${RL_TP_SIZE:-1}"          # vLLM tensor-parallel; 1.5B fits per-GPU, so 1
# For newly recomputed gradients, NUM_PARALLEL is the recompute fan-out. For scoring
# precomputed gradients, set it to the existing shard count instead.
export NUM_PARALLEL="${NUM_PARALLEL:-2}"
export RL_GPU_MEMORY_UTILIZATION="${RL_GPU_MEMORY_UTILIZATION:-0.6}"
# Conservative micro-batches for 24GB — starting points, raise them if VRAM allows.
export RL_PPO_MICRO_BATCH_SIZE_PER_GPU="${RL_PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
export RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
export RL_USE_WANDB="${RL_USE_WANDB:-0}"      # 1 -> also log to wandb (needs WANDB_API_KEY)

# -----------------------------------------------------------------------------
# Runtime hygiene
# -----------------------------------------------------------------------------
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# -----------------------------------------------------------------------------
# Create output dirs (idempotent)
# -----------------------------------------------------------------------------
mkdir -p "$CROPI_WORK" "$CROPI_WORK/venvs" "$DATA_ROOT" "$MODELS_DIR" "$CKPT_ROOT" \
         "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$UV_CACHE_DIR" \
         "$CROPI_REPO/logs" 2>/dev/null || true

_cropi_write_probe="$CROPI_WORK/.cropi_write_probe.$$"
if { : > "$_cropi_write_probe"; } 2>/dev/null; then
  rm -f "$_cropi_write_probe"
  export CROPI_WORK_WRITABLE=1
else
  export CROPI_WORK_WRITABLE=0
fi

# -----------------------------------------------------------------------------
# venv activation helper:  cropi_activate
# Activates the uv-managed cropi env so BOTH `python ...` (split_files.py,
# model_merger.py) and `uv run ...` inside cropi/scripts/* use the same interpreter.
# -----------------------------------------------------------------------------
cropi_activate() {
  if [ ! -f "$CROPI_VENV/bin/activate" ]; then
    echo "[cropi_activate] cropi venv not found at $CROPI_VENV" >&2
    echo "[cropi_activate] create it:  bash scripts/install.sh cropi" >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "$CROPI_VENV/bin/activate"
  echo "[cropi_activate] cropi venv active: $(command -v python)"
}
if [ -n "${BASH_VERSION:-}" ]; then export -f cropi_activate; fi

# -----------------------------------------------------------------------------
# Existence checks (warn-only — never aborts)
# -----------------------------------------------------------------------------
_cropi_missing=0
_cropi_warn() {
  local var="$1" path="$2" fix="$3"
  if [ ! -e "$path" ]; then
    if [ "$_cropi_missing" = "0" ]; then
      echo ""
      echo "------------------------------------------------------------------"
      echo "[setup_env] WARNINGS: the following paths do not exist yet."
      echo "------------------------------------------------------------------"
    fi
    printf "  [missing] %-18s %s\n" "$var" "$path"
    printf "            fix:  %s\n" "$fix"
    _cropi_missing=$((_cropi_missing + 1))
  fi
}
_cropi_warn_custom() {
  local label="$1" detail="$2" fix="$3"
  if [ "$_cropi_missing" = "0" ]; then
    echo ""
    echo "------------------------------------------------------------------"
    echo "[setup_env] WARNINGS: the following paths do not exist yet."
    echo "------------------------------------------------------------------"
  fi
  printf "  [warning] %-18s %s\n" "$label" "$detail"
  printf "            fix:  %s\n" "$fix"
  _cropi_missing=$((_cropi_missing + 1))
}
_cropi_warn GROUP_VOLUME "$GROUP_VOLUME" "mount it, or: export GROUP_VOLUME=/your/large/mount (before sourcing)"
if [ "$CROPI_WORK_WRITABLE" != "1" ]; then
  _cropi_warn_custom CROPI_WORK "$CROPI_WORK is not writable" "mount/write-enable it, or export CROPI_WORK=/your/writable/workspace before sourcing"
fi
_cropi_warn CROPI_VENV   "$CROPI_VENV"   "bash scripts/install.sh cropi"
_cropi_warn VERL_VENV    "$VERL_VENV"    "bash scripts/install.sh verl   (or point RL_PYTHON at your verl env)"
_cropi_warn BASE_MODEL_PATH "$BASE_MODEL_PATH" "download $HFID_BASE_MODEL into it (see SETUP.md)"

if [ "$_cropi_missing" -gt 0 ]; then
  echo "------------------------------------------------------------------"
  echo "[setup_env] $_cropi_missing path(s) missing — env vars are still exported."
  echo "------------------------------------------------------------------"
else
  echo "[setup_env] All paths verified ✓"
fi
echo ""
echo "CROPI env loaded  (VM defaults; RL GPU count auto-detected when possible)."
echo "  CROPI_WORK       = $CROPI_WORK"
echo "  DATA_ROOT        = $DATA_ROOT"
echo "  BASE_MODEL_PATH  = $BASE_MODEL_PATH"
echo "  CKPT_ROOT        = $CKPT_ROOT"
echo "  cropi venv       = $CROPI_VENV   (UV_PROJECT_ENVIRONMENT)"
echo "  RL_PYTHON        = $RL_PYTHON"
echo "  RL_NUM_GPUS=$RL_NUM_GPUS  RL_TP_SIZE=$RL_TP_SIZE  NUM_PARALLEL=$NUM_PARALLEL"
echo "  detected GPUs    = $CROPI_DETECTED_GPUS"
echo "  activate cropi   = cropi_activate"

unset -f _cropi_warn
unset -f _cropi_warn_custom
unset _cropi_missing
unset _cropi_detected_gpus
unset _cropi_write_probe
