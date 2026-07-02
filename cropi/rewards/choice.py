"""Multiple-choice (A/B/C/D) reward for MMLU-style RLVR.

Used in three places:
  - cropi/inference/generate_rollouts.py  (offline rollout rewards)
  - cropi/eval/eval_math.py                (accuracy eval)
  - verl RL training, via `custom_reward_function.path=<this file> .name=compute_score`

Reward = 1.0 if the model's boxed letter matches the gold letter, else 0.0.
"""
import re

_BOX = re.compile(r"\\boxed\{([^}]*)\}")


def extract_choice(text: str) -> str:
    """Pull an A/B/C/D letter from a completion, preferring the last \\boxed{}."""
    text = text or ""
    for span in reversed(_BOX.findall(text)):
        for ch in span.strip().upper():
            if ch in "ABCD":
                return ch
    # fallback: last standalone A-D token (e.g. "answer is C")
    hits = re.findall(r"\b([ABCD])\b", text.upper())
    return hits[-1] if hits else ""


def choice_reward(response: str, gold: str) -> float:
    return 1.0 if extract_choice(response) == str(gold).strip().upper() else 0.0


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None):
    """verl-compatible entrypoint (custom_reward_function.name=compute_score)."""
    return choice_reward(solution_str, ground_truth)
