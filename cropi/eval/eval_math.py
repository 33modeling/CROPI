"""Evaluate a checkpoint on a test parquet (gsm8k/mmlu-boxed) and report accuracy.

Greedy single-sample decode + math-verify (same reward as training). Writes a
small JSON summary so the full-vs-selection arms can be compared directly.

    $RL_PYTHON cropi/eval/eval_math.py \
        --parquet $DATA_ROOT/gsm8k/test_qwen.parquet \
        --model   <ckpt>/huggingface \
        --tag     gsm8k_full \
        --out     $RESULTS_DIR/gsm8k_full.json
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from cropi.inference.generate_rollouts import math_reward  # reuse the reward


def parse_args():
    ap = argparse.ArgumentParser(description="Accuracy eval for a checkpoint.")
    ap.add_argument("--parquet", required=True, help="test_qwen.parquet")
    ap.add_argument("--model", required=True, help="HF checkpoint dir (…/actor/huggingface)")
    ap.add_argument("--tag", required=True, help="label for this run (e.g. gsm8k_full)")
    ap.add_argument("--out", required=True, help="destination JSON summary")
    ap.add_argument("--max_tokens", type=int, default=int(os.environ.get("RL_MAX_RESPONSE_LENGTH", "2048")))
    ap.add_argument("--max_prompt_tokens", type=int, default=1024)
    ap.add_argument("--tp_size", type=int, default=int(os.environ.get("RL_TP_SIZE", "1")))
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.out) and not args.overwrite:
        print(f"[skip] {args.out} exists:\n{open(args.out).read()}")
        return
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    rows = df.to_dict(orient="records")
    if args.limit > 0:
        rows = rows[: args.limit]

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp_size,
              gpu_memory_utilization=args.gpu_mem, trust_remote_code=True,
              max_model_len=args.max_prompt_tokens + args.max_tokens)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_tokens)

    prompts, golds = [], []
    for r in rows:
        prompts.append(tok.apply_chat_template(list(r["prompt"]), tokenize=False, add_generation_prompt=True))
        golds.append(str(r.get("answer") or r["reward_model"]["ground_truth"]))

    outs = llm.generate(prompts, sp)
    correct = sum(math_reward(o.outputs[0].text, g) for o, g in zip(outs, golds))
    acc = correct / max(1, len(golds))
    summary = {"tag": args.tag, "model": args.model, "n": len(golds),
               "correct": correct, "accuracy": acc}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] {args.tag}: accuracy={acc:.4f} ({int(correct)}/{len(golds)}) -> {args.out}")


if __name__ == "__main__":
    main()
