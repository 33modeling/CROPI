"""gsm8k numeric reward for verl's custom_reward_function.

verl 0.4.1's built-in reward only maps data_source=="openai/gsm8k" (ours is "gsm8k")
and defaults to strict "#### N" extraction, which our qwen25-math-cot prompt does not
emit — so it would score every rollout 0. This reward extracts the LAST number in the
completion (flexible) and compares numerically to the gold answer, so it works with the
CROPI/Qwen prompt format. Mirrors the choice.py interface used for MMLU.

Wire via:
  CUSTOM_REWARD_PATH=cropi/rewards/gsm8k_math.py CUSTOM_REWARD_NAME=compute_score
"""

import re

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_last_number(text: str):
    # prefer an explicit \boxed{...} or "#### N" if present, else the last number
    for pat in (r"\\boxed\{([^}]*)\}", r"####\s*(-?[0-9\.,]+)"):
        m = re.findall(pat, text)
        if m:
            got = _NUM.findall(m[-1])
            if got:
                return got[-1]
    nums = _NUM.findall(text)
    for tok in reversed(nums):
        tok = tok.rstrip(".")
        if tok not in ("", ".", "-"):
            return tok
    return None


def _norm(s: str):
    return str(s).replace(",", "").replace("$", "").strip().rstrip(".")


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None):
    pred = extract_last_number(solution_str or "")
    if pred is None:
        return 0.0
    p, g = _norm(pred), _norm(ground_truth)
    try:
        return 1.0 if abs(float(p) - float(g)) < 1e-6 else 0.0
    except ValueError:
        return 1.0 if p == g else 0.0
