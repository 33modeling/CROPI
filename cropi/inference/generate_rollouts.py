"""Generate off-policy rollouts for CROPI scoring (replaces the external
Qwen2.5-Math eval dependency).

For each prompt in a parquet it samples N completions with vLLM, scores each
with math-verify, and writes one JSONL line:

    {"prompt": <user content str>, "answer": <gold>,
     "responses": [str, ...], "rewards": [float, ...]}

Key contract:
  - "prompt" is prompt[1]["content"] (the user turn) so it matches the key that
    compute_inf_score / select use against the parquet.
  - The model still SEES the chat-templated prompt during generation.
  - Output path/name follow run_cropi.sh's <infer_note> convention exactly.

Run in the verl env (it has vLLM):
    $RL_PYTHON cropi/inference/generate_rollouts.py \
        --parquet $DATA_ROOT/gsm8k/train_qwen.parquet \
        --model   $BASE_MODEL_PATH \
        --output  $DATA_ROOT/gsm8k/<model_name>/train_<infer_note>.jsonl \
        --n 8 --temperature 0.5 --max_tokens 2048

Idempotent: prompts already present (with a non-empty response list) in the
output are skipped, so re-running resumes / no-ops.
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="vLLM rollout generator for CROPI.")
    ap.add_argument("--parquet", required=True, help="train_qwen.parquet / valid_qwen.parquet")
    ap.add_argument("--model", required=True, help="policy model path (base or exported actor)")
    ap.add_argument("--output", required=True, help="destination JSONL (train_<infer_note>.jsonl)")
    ap.add_argument("--n", type=int, default=8, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--max_prompt_tokens", type=int, default=1024)
    ap.add_argument("--tp_size", type=int, default=int(os.environ.get("RL_TP_SIZE", "1")))
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward", choices=["auto", "math", "choice"], default="auto",
                    help="reward type; auto -> infer from parquet data_source (mmlu=choice)")
    ap.add_argument("--limit", type=int, default=-1, help="debug: cap #prompts")
    return ap.parse_args()


def pick_reward(name: str, data_source: str = ""):
    if name == "choice" or (name == "auto" and "mmlu" in (data_source or "").lower()):
        from cropi.rewards.choice import choice_reward
        return choice_reward
    return math_reward


def math_reward(response: str, gold: str) -> float:
    """1.0 if the boxed/parsed answer matches gold, else 0.0 (mirrors rl_utils)."""
    try:
        from math_verify.metric import math_metric
        from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
        verify = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
        try:
            score, _ = verify(["\\boxed{" + str(gold) + "}"], [response])
            return float(score)
        except Exception:
            return 0.0
    except ImportError:
        # fallback: exact match on the last boxed span
        import re
        m = re.findall(r"\\boxed\{([^}]*)\}", response)
        pred = (m[-1] if m else "").strip().replace(",", "").replace("$", "")
        return 1.0 if pred == str(gold).strip() else 0.0


def load_prompts(parquet: str):
    df = pd.read_parquet(parquet)
    rows = []
    for r in df.to_dict(orient="records"):
        chat = list(r["prompt"])
        user = chat[1]["content"]
        gold = r.get("answer")
        if gold is None:
            gold = r["reward_model"]["ground_truth"]
        rows.append({"chat": chat, "user": user, "gold": str(gold),
                     "data_source": r.get("data_source", "")})
    return rows


def already_done(output: str) -> set[str]:
    done = set()
    if os.path.exists(output):
        with open(output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if item.get("responses"):
                    done.add(item["prompt"])
    return done


def main():
    args = parse_args()
    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)

    rows = load_prompts(args.parquet)
    if args.limit > 0:
        rows = rows[: args.limit]
    reward_fn = pick_reward(args.reward, rows[0]["data_source"] if rows else "")
    print(f"[rollout] reward={reward_fn.__name__} (--reward={args.reward})")
    done = already_done(args.output)
    todo = [r for r in rows if r["user"] not in done]
    print(f"[rollout] {len(rows)} prompts, {len(done)} already done, {len(todo)} to generate")
    if not todo:
        print("[rollout] nothing to do (resume complete).")
        return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp_size,
              gpu_memory_utilization=args.gpu_mem, trust_remote_code=True,
              max_model_len=args.max_prompt_tokens + args.max_tokens, seed=args.seed)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed)

    templated = [tok.apply_chat_template(r["chat"], tokenize=False, add_generation_prompt=True)
                 for r in todo]

    # append-only so a crash mid-run still leaves a resumable file
    with open(args.output, "a", encoding="utf-8") as fout:
        outs = llm.generate(templated, sp)
        for r, out in zip(todo, outs):
            responses = [o.text for o in out.outputs]
            rewards = [reward_fn(resp, r["gold"]) for resp in responses]
            fout.write(json.dumps({
                "prompt": r["user"],
                "answer": r["gold"],
                "responses": responses,
                "rewards": rewards,
            }, ensure_ascii=False) + "\n")
            fout.flush()
    print(f"[rollout] wrote {len(todo)} prompts -> {args.output}")


if __name__ == "__main__":
    main()
