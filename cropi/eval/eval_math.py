"""Evaluate a checkpoint on a test parquet (gsm8k/mmlu-boxed) and report accuracy.

Greedy single-sample decode + the same reward used in training. Besides the
headline accuracy, the JSON summary keeps a per-example record (gold, extracted
answer, correct flag, response length, truncation) plus decode/length metadata,
so make_report.py can line the full-vs-CROPI arms up example by example and
write a detailed report.

    $RL_PYTHON cropi/eval/eval_math.py \
        --parquet $DATA_ROOT/gsm8k/test_qwen.parquet \
        --model   <ckpt>/huggingface \
        --tag     gsm8k_full \
        --out     $RESULTS_DIR/gsm8k_full.json
"""
import argparse
import datetime
import json
import os
import re
import statistics
from pathlib import Path

import pandas as pd

from cropi.inference.generate_rollouts import math_reward, pick_reward  # reuse rewards


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
    ap.add_argument("--reward", choices=["auto", "math", "choice"], default="auto")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def _extract_pred(text: str, reward_name: str) -> str:
    """Best-effort answer string for the report. Correctness still comes from the
    reward function; this is only what we *show* next to gold in the writeup."""
    if reward_name == "choice_reward":
        try:
            from cropi.rewards.choice import extract_choice
            return extract_choice(text)
        except Exception:
            pass
    m = re.findall(r"\\boxed\{([^}]*)\}", text or "")
    return m[-1].strip() if m else ""


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

    data_source = str(rows[0].get("data_source", "")) if rows else ""
    reward_fn = pick_reward(args.reward, data_source)
    print(f"[eval] reward={reward_fn.__name__}")

    prompts, golds, questions = [], [], []
    for r in rows:
        prompts.append(tok.apply_chat_template(list(r["prompt"]), tokenize=False, add_generation_prompt=True))
        golds.append(str(r.get("answer") or r["reward_model"]["ground_truth"]))
        try:
            questions.append(str(list(r["prompt"])[-1]["content"]))
        except Exception:
            questions.append("")

    outs = llm.generate(prompts, sp)

    examples, resp_tokens = [], []
    correct = 0.0
    n_trunc = 0
    for i, (o, g, q) in enumerate(zip(outs, golds, questions)):
        out0 = o.outputs[0]
        c = float(reward_fn(out0.text, g))
        correct += c
        ntok = len(out0.token_ids)
        resp_tokens.append(ntok)
        truncated = (getattr(out0, "finish_reason", None) == "length")
        n_trunc += int(truncated)
        examples.append({
            "idx": i,
            "gold": g,
            "pred": _extract_pred(out0.text, reward_fn.__name__),
            "correct": bool(c >= 0.5),
            "resp_tokens": ntok,
            "truncated": truncated,
            "question": q[:200],
        })

    n = len(golds)
    acc = correct / max(1, n)
    summary = {
        "tag": args.tag,
        "model": args.model,
        "dataset": data_source or args.tag,
        "reward": reward_fn.__name__,
        "n": n,
        "correct": int(correct),
        "accuracy": acc,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "decode": {"temperature": 0.0, "max_tokens": args.max_tokens,
                   "max_prompt_tokens": args.max_prompt_tokens},
        "length": {
            "mean_tokens": round(statistics.fmean(resp_tokens), 1) if resp_tokens else 0.0,
            "max_tokens": max(resp_tokens) if resp_tokens else 0,
            "truncated": n_trunc,
        },
        "examples": examples,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[eval] {args.tag}: accuracy={acc:.4f} ({int(correct)}/{n}) -> {args.out}")


if __name__ == "__main__":
    main()
