#!/usr/bin/env bash
# Surfaces everything that can't be known until you're on the A100 box:
#   - paths (datasets, model, workspace) exist / writable
#   - GPU count matches NUM_PARALLEL / RL_NUM_GPUS
#   - Qwen3.5-9B chat template + whether it is a THINKING model (-> response length)
#   - raw dataset format under $IT_DATASETS (parquet / jsonl / arrow / HF dir)
#   - the cropi venv can actually load the 9B weights (transformers<5 pin check)
# Read-only and safe to re-run. Run this FIRST:  bash scripts/preflight.sh
set -uo pipefail

: "${IT_DATASETS:?source scripts/setup_env_a100.sh first}"
: "${BASE_MODEL_PATH:?source scripts/setup_env_a100.sh first}"
: "${CROPI_WORK:?source scripts/setup_env_a100.sh first}"

ok(){   echo "  [ OK ] $*"; }
warn(){ echo "  [WARN] $*"; }
bad(){  echo "  [FAIL] $*"; }

echo "=============================================================="
echo " CROPI preflight"
echo "=============================================================="

echo "-- paths --------------------------------------------------"
for p in "$IT_DATASETS/gsm8k" "$IT_DATASETS/mmlu" "$BASE_MODEL_PATH"; do
  [ -e "$p" ] && ok "exists: $p" || bad "MISSING: $p"
done
mkdir -p "$CROPI_WORK" 2>/dev/null && [ -w "$CROPI_WORK" ] \
  && ok "workspace writable: $CROPI_WORK" || bad "cannot write workspace: $CROPI_WORK"

echo "-- gpus ---------------------------------------------------"
if command -v nvidia-smi >/dev/null 2>&1; then
  n=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  ok "visible GPUs: $n"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/       /'
  [ "$n" = "${NUM_PARALLEL:-}" ] && ok "NUM_PARALLEL($NUM_PARALLEL) matches GPU count" \
    || bad "NUM_PARALLEL(${NUM_PARALLEL:-unset}) != GPU count($n) -> get-grad will target missing devices"
else
  bad "nvidia-smi not found"
fi

echo "-- raw dataset format -------------------------------------"
for d in gsm8k mmlu; do
  echo "  [$d] $(ls "$IT_DATASETS/$d" 2>/dev/null | tr '\n' ' ')"
done

echo "-- model config / THINKING check --------------------------"
python3 - "$BASE_MODEL_PATH" <<'PY'
import json, os, sys, glob
mp = sys.argv[1]
def load(name):
    p = os.path.join(mp, name)
    if os.path.exists(p):
        try: return json.load(open(p))
        except Exception as e: return {"_error": str(e)}
    return None
cfg = load("config.json") or {}
print("       architectures :", cfg.get("architectures"))
print("       model_type    :", cfg.get("model_type"))
print("       max_pos_embed :", cfg.get("max_position_embeddings"))
n_st = len(glob.glob(os.path.join(mp, "*.safetensors")))
print(f"       safetensors   : {n_st} shard(s)")
tconf = load("tokenizer_config.json") or {}
tmpl = tconf.get("chat_template") or ""
if not tmpl:
    # newer HF ships chat_template.jinja separately
    jp = os.path.join(mp, "chat_template.jinja")
    if os.path.exists(jp): tmpl = open(jp).read()
markers = [m for m in ("<think>", "</think>", "thinking", "enable_thinking", "reasoning") if m in tmpl.lower()]
if markers:
    print("       THINKING model? LIKELY  (template markers:", markers, ")")
    print("       -> raise RL_MAX_RESPONSE_LENGTH (e.g. 4096-8192) before running.")
else:
    print("       THINKING model? probably NOT (no think markers in chat_template)")
gen = load("generation_config.json") or {}
print("       eos_token_id  :", gen.get("eos_token_id", cfg.get("eos_token_id")))
PY

echo "-- cropi venv can load 9B? (transformers<5 pin) -----------"
if [ -n "${CROPI_VENV:-}" ] && [ -f "$CROPI_VENV/bin/python" ]; then
  "$CROPI_VENV/bin/python" - "$BASE_MODEL_PATH" <<'PY'
import sys
try:
    import transformers
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
    print(f"  [ OK ] transformers {transformers.__version__} parsed config ({type(c).__name__})")
except Exception as e:
    print(f"  [FAIL] cropi env cannot parse Qwen3.5 config: {e}")
    print("         -> may need a newer transformers in the cropi venv for get-grad.")
PY
else
  warn "CROPI_VENV not built yet (bash scripts/install.sh cropi)"
fi

echo "-- verl interpreter present? ------------------------------"
if [ -n "${RL_PYTHON:-}" ] && command -v "$RL_PYTHON" >/dev/null 2>&1; then
  "$RL_PYTHON" - <<'PY'
mods = []
for m in ("verl","vllm","transformers","torch"):
    try:
        mod = __import__(m); mods.append(f"{m}={getattr(mod,'__version__','?')}")
    except Exception as e:
        mods.append(f"{m}=MISSING({e.__class__.__name__})")
print("       " + "  ".join(mods))
print("       -> confirm this vllm/transformers supports the Qwen3.5 architecture.")
PY
else
  warn "RL_PYTHON not found: ${RL_PYTHON:-unset} (bash scripts/install.sh verl)"
fi

echo "=============================================================="
echo " Review any [FAIL]/[WARN] above before running scripts/run_gsm8k.sh"
echo "=============================================================="
