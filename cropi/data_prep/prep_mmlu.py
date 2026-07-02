"""Preprocess MMLU into the parquet layout CROPI + verl expect (choice task).

Reads raw MMLU from $IT_DATASETS/mmlu (auto-detects layout; HF `cais/mmlu` ships
an `all/` config dir with auxiliary_train / validation / dev / test parquets) and
writes under $DATA_ROOT/mmlu/:

    train_qwen.parquet   train pool  (auxiliary_train if present, else carved test)
    valid_qwen.parquet   CROPI influence target (from validation)
    test_qwen.parquet    evaluation set (test)

Each prompt shows the question + A/B/C/D options; the gold answer is the letter,
scored by cropi.rewards.choice (exact letter match, verifiable → RLVR-ready).

Idempotent: skips if all three parquets exist (unless --overwrite).
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = ("Answer the following multiple choice question. "
                 "Choose the single best option and put its letter "
                 "(A, B, C, or D) within \\boxed{}.")


def parse_args():
    ap = argparse.ArgumentParser(description="Preprocess MMLU for CROPI/verl (choice task).")
    ap.add_argument("--raw_dir", default=os.environ.get("IT_DATASETS", "") + "/mmlu")
    ap.add_argument("--out_dir", default=os.environ.get("DATA_ROOT", "./data") + "/mmlu")
    ap.add_argument("--data_source", default="mmlu")
    ap.add_argument("--n_train", type=int, default=8000, help="cap train pool (rollouts get expensive); <=0 = all")
    ap.add_argument("--n_valid", type=int, default=500, help="CROPI influence-target size")
    ap.add_argument("--n_test", type=int, default=2000, help="eval size; <=0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def _read_any(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_split(raw_dir: Path, names: list[str]) -> list[dict]:
    """Find rows for the first split name that matches, trying common layouts."""
    # HF load_from_disk
    if (raw_dir / "dataset_dict.json").exists():
        from datasets import load_from_disk
        ds = load_from_disk(str(raw_dir))
        for n in names:
            if n in ds:
                return list(ds[n])
    # parquet/jsonl by name, in raw_dir, raw_dir/all, raw_dir/<name>
    roots = [raw_dir, raw_dir / "all", raw_dir / "data"]
    for n in names:
        for root in roots + [raw_dir / n]:
            if not root.exists():
                continue
            for ext in ("parquet", "jsonl", "json"):
                hits = sorted(glob.glob(str(root / f"{n}*.{ext}"))) or \
                       sorted(glob.glob(str(root / f"*{n}*.{ext}")))
                if hits:
                    return [r for h in hits for r in _read_any(Path(h))]
        # a bare split subdir of parquets: raw_dir/<name>/*.parquet
        sub = raw_dir / n
        if sub.is_dir():
            hits = sorted(glob.glob(str(sub / "*.parquet"))) or sorted(glob.glob(str(sub / "*.jsonl")))
            if hits:
                return [r for h in hits for r in _read_any(Path(h))]
    return []


def get_choices(row: dict) -> list[str]:
    if "choices" in row and row["choices"] is not None:
        return list(row["choices"])
    # some dumps use A/B/C/D columns
    if all(k in row for k in ("A", "B", "C", "D")):
        return [row["A"], row["B"], row["C"], row["D"]]
    raise KeyError(f"no choices in row keys={list(row)}")


def get_gold_letter(row: dict) -> str:
    a = row.get("answer", row.get("label"))
    if isinstance(a, (int,)) or (isinstance(a, str) and a.isdigit()):
        return "ABCD"[int(a)]
    s = str(a).strip().upper()
    return s[0] if s and s[0] in "ABCD" else s


def get_question(row: dict) -> str:
    for k in ("question", "query", "problem"):
        if k in row and isinstance(row[k], str):
            return row[k].strip()
    raise KeyError(f"no question in row keys={list(row)}")


def to_records(rows: list[dict], data_source: str, split: str) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        try:
            q = get_question(row)
            ch = get_choices(row)
            gold = get_gold_letter(row)
        except KeyError:
            continue
        if len(ch) < 2 or gold not in "ABCD":
            continue
        body = q + "\n" + "\n".join(f"{'ABCD'[j]}. {c}" for j, c in enumerate(ch[:4]))
        out.append({
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": body},
            ],
            "ability": "mcq",
            "reward_model": {"style": "rule", "ground_truth": gold},
            "answer": gold,
            "extra_info": {"split": split, "index": i},
        })
    return out


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n and n > 0 and len(df) > n:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def main():
    args = parse_args()
    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {s: out_dir / f"{s}_qwen.parquet" for s in ("train", "valid", "test")}
    if all(p.exists() for p in paths.values()) and not args.overwrite:
        for s, p in paths.items():
            print(f"[skip] {s}: {p} ({len(pd.read_parquet(p))} rows)")
        return

    aux = find_split(raw_dir, ["auxiliary_train", "train"])
    val = find_split(raw_dir, ["validation", "val"])
    test = find_split(raw_dir, ["test"])
    if not test:
        raise FileNotFoundError(f"no test split under {raw_dir} (contents: {[p.name for p in raw_dir.iterdir()]})")

    df_test_all = pd.DataFrame(to_records(test, args.data_source, "test")).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    if aux:  # canonical: train=auxiliary_train, valid=validation, test=test
        df_train = _sample(pd.DataFrame(to_records(aux, args.data_source, "train")), args.n_train, args.seed)
        src = val if val else test
        df_valid = _sample(pd.DataFrame(to_records(src, args.data_source, "valid")), args.n_valid, args.seed)
        df_test = _sample(df_test_all, args.n_test, args.seed)
        print(f"[info] using auxiliary_train as train pool ({len(df_train)}), validation as valid target")
    else:    # fallback: carve disjoint train/valid/test from the test split
        n_valid = min(args.n_valid, len(df_test_all) // 10)
        n_test = args.n_test if args.n_test > 0 else len(df_test_all) // 5
        df_valid = df_test_all.iloc[:n_valid].reset_index(drop=True)
        df_test = df_test_all.iloc[n_valid:n_valid + n_test].reset_index(drop=True)
        df_train = _sample(df_test_all.iloc[n_valid + n_test:].reset_index(drop=True), args.n_train, args.seed)
        print(f"[info] no auxiliary_train; carved disjoint train/valid/test from test split")

    df_train.to_parquet(paths["train"])
    df_valid.to_parquet(paths["valid"])
    df_test.to_parquet(paths["test"])
    print(f"[done] train={len(df_train)}  valid={len(df_valid)}  test={len(df_test)} -> {out_dir}")


if __name__ == "__main__":
    main()
