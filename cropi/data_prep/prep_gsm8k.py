"""Preprocess gsm8k into the parquet layout CROPI + verl expect.

Reads the raw dataset from $IT_DATASETS/gsm8k (auto-detects parquet / jsonl /
HuggingFace arrow dir) and writes, under $DATA_ROOT/gsm8k/:

    train_qwen.parquet   train pool minus the held-out valid slice
    valid_qwen.parquet   CROPI influence target (hold-out from train)
    test_qwen.parquet    final evaluation set (official gsm8k test)

Schema (one row per prompt), compatible with:
  - CROPI  : prompt[1]["content"] is used as the match key (compute_inf_score / select)
  - verl   : data_source / prompt / reward_model.ground_truth / extra_info

Idempotent: if all three parquets already exist it exits without rewriting
(pass --overwrite to force).
"""
import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_args():
    ap = argparse.ArgumentParser(description="Preprocess gsm8k for CROPI/verl.")
    ap.add_argument("--raw_dir", default=os.environ.get("IT_DATASETS", "") + "/gsm8k",
                    help="Raw gsm8k dir (default: $IT_DATASETS/gsm8k)")
    ap.add_argument("--out_dir", default=os.environ.get("DATA_ROOT", "./data") + "/gsm8k",
                    help="Output dir (default: $DATA_ROOT/gsm8k)")
    ap.add_argument("--data_source", default="gsm8k", help="verl reward-manager key")
    ap.add_argument("--n_valid", type=int, default=500, help="hold-out valid size (from train)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def _read_any(path: Path) -> list[dict]:
    """Read one file into a list of row dicts (parquet / json / jsonl)."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if path.suffix in (".jsonl", ".json"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        if not txt:
            return rows
        if path.suffix == ".jsonl" or txt[0] != "[":
            for line in txt.splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        else:
            rows = json.loads(txt)
        return rows
    raise ValueError(f"unsupported file type: {path}")


def load_split(raw_dir: Path, split: str) -> list[dict]:
    """Find the raw file(s) for a split. Handles parquet/jsonl/arrow-dir layouts."""
    # 1) HuggingFace load_from_disk / arrow dir
    if (raw_dir / "dataset_dict.json").exists() or (raw_dir / split / "dataset_info.json").exists():
        from datasets import load_from_disk  # lazy import
        ds = load_from_disk(str(raw_dir))
        key = split if split in ds else ("test" if split == "test" else "train")
        return list(ds[key])
    # 2) flat files named after the split
    for pat in (f"{split}*.parquet", f"{split}*.jsonl", f"{split}*.json",
                f"*{split}*.parquet", f"*{split}*.jsonl"):
        hits = sorted(raw_dir.glob(pat))
        if hits:
            rows: list[dict] = []
            for h in hits:
                rows.extend(_read_any(h))
            return rows
    # 3) HF hub "main" config subdir (openai/gsm8k -> main/train-*.parquet)
    for sub in ("main", "socratic", "."):
        hits = sorted((raw_dir / sub).glob(f"{split}*.parquet")) if (raw_dir / sub).exists() else []
        if hits:
            return [r for h in hits for r in _read_any(h)]
    raise FileNotFoundError(f"could not locate '{split}' split under {raw_dir}. "
                            f"Contents: {[p.name for p in raw_dir.iterdir()]}")


def get_question(row: dict) -> str:
    for k in ("question", "problem", "query", "input", "prompt"):
        if k in row and isinstance(row[k], str):
            return row[k].strip()
    raise KeyError(f"no question field in row keys={list(row)}")


_ANS_RE = re.compile(r"####\s*(.+)")


def get_answer(row: dict) -> str:
    """gsm8k gold answer is the number after '####'. Falls back to answer/target fields."""
    for k in ("answer", "solution", "target", "label"):
        if k in row and isinstance(row[k], str):
            m = _ANS_RE.search(row[k])
            ans = (m.group(1) if m else row[k]).strip()
            return ans.replace(",", "").replace("$", "").strip()
    raise KeyError(f"no answer field in row keys={list(row)}")


def to_records(rows: list[dict], data_source: str, split: str) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        q = get_question(row)
        a = get_answer(row)
        out.append({
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": a},
            "answer": a,                       # convenience field for our own tooling
            "extra_info": {"split": split, "index": i},
        })
    return out


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {s: out_dir / f"{s}_qwen.parquet" for s in ("train", "valid", "test")}
    if all(p.exists() for p in paths.values()) and not args.overwrite:
        for s, p in paths.items():
            print(f"[skip] {s}: {p} exists ({len(pd.read_parquet(p))} rows)")
        return

    train_rows = to_records(load_split(raw_dir, "train"), args.data_source, "train")
    test_rows = to_records(load_split(raw_dir, "test"), args.data_source, "test")

    # deterministic hold-out valid slice carved from train
    df_train = pd.DataFrame(train_rows).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_valid = min(args.n_valid, max(1, len(df_train) // 10))
    df_valid = df_train.iloc[:n_valid].reset_index(drop=True)
    df_train = df_train.iloc[n_valid:].reset_index(drop=True)
    df_test = pd.DataFrame(test_rows)

    df_train.to_parquet(paths["train"])
    df_valid.to_parquet(paths["valid"])
    df_test.to_parquet(paths["test"])
    print(f"[done] train={len(df_train)}  valid={len(df_valid)}  test={len(df_test)}")
    print(f"[done] wrote -> {out_dir}")


if __name__ == "__main__":
    main()
