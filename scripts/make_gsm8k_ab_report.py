#!/usr/bin/env python3
"""Build an HTML report for the local GSM8K Arm A vs Arm B run."""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--timings", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--a-result", required=True)
    p.add_argument("--b-result", required=True)
    p.add_argument("--a-ckpt-root", required=True)
    p.add_argument("--b-ckpt-root", required=True)
    p.add_argument("--b-data-dir", required=True)
    p.add_argument("--eval-parquet", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--repo", required=True)
    return p.parse_args()


def read_json(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def fmt_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = int(round(float(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def read_timings(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("phase\t"):
            continue
        parts = line.split("\t")
        if len(parts) != 6:
            continue
        phase, status, start_iso, end_iso, seconds, log_path = parts
        try:
            sec_value = float(seconds)
        except ValueError:
            sec_value = None
        rows.append(
            {
                "phase": phase,
                "status": status,
                "start": start_iso,
                "end": end_iso,
                "seconds": sec_value,
                "log": log_path,
            }
        )
    return rows


def phase_seconds(rows: list[dict], *names: str) -> float | None:
    total = 0.0
    seen = False
    for row in rows:
        if row["phase"] in names and row["seconds"] is not None:
            total += row["seconds"]
            seen = True
    return total if seen else None


def checkpoint_times(ckpt_root: str | Path) -> list[tuple[int, float, str]]:
    root = Path(ckpt_root)
    out = []
    for data_pt in root.glob("global_step_*/data.pt"):
        try:
            step = int(data_pt.parent.name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        out.append((step, data_pt.stat().st_mtime, str(data_pt)))
    return sorted(out)


def estimate_a_training_seconds(a_root: str | Path) -> tuple[float | None, str]:
    points = checkpoint_times(a_root)
    if len(points) < 2:
        return None, "No estimate: fewer than two checkpoints."
    first_step, first_ts, _ = points[0]
    last_step, last_ts, _ = points[-1]
    if last_step <= first_step:
        return None, "No estimate: checkpoint step order is invalid."
    seconds_per_step = (last_ts - first_ts) / (last_step - first_step)
    estimate = seconds_per_step * last_step
    start_est = datetime.fromtimestamp(last_ts - estimate).isoformat(timespec="seconds")
    end = datetime.fromtimestamp(last_ts).isoformat(timespec="seconds")
    note = (
        f"Estimated from checkpoint timestamps step {first_step} -> {last_step}; "
        f"estimated start {start_est}, final checkpoint {end}."
    )
    return estimate, note


def newest_selection_stat(b_data_dir: str | Path) -> tuple[dict | None, str | None]:
    root = Path(b_data_dir)
    candidates = sorted(root.glob("*ratio0.1*iter0_stat.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None, None
    p = candidates[-1]
    return read_json(p), str(p)


def result_row(label: str, data: dict | None) -> str:
    if data is None:
        return (
            f"<tr><td>{html.escape(label)}</td><td colspan='4'>missing</td></tr>"
        )
    return (
        "<tr>"
        f"<td>{html.escape(label)}</td>"
        f"<td><code>{html.escape(data.get('tag', ''))}</code></td>"
        f"<td>{fmt_pct(data.get('accuracy'))}</td>"
        f"<td>{int(data.get('correct', 0))}/{int(data.get('n', 0))}</td>"
        f"<td><code>{html.escape(data.get('model', ''))}</code></td>"
        "</tr>"
    )


def main() -> None:
    args = parse_args()
    a = read_json(args.a_result)
    b = read_json(args.b_result)
    timings = read_timings(args.timings)
    a_train_est, a_train_note = estimate_a_training_seconds(args.a_ckpt_root)
    b_pipeline = phase_seconds(
        timings,
        "b_rollout_train",
        "b_rollout_valid",
        "b_grad",
        "b_score_select_train",
    )
    b_train = phase_seconds(timings, "b_score_select_train")
    eval_total = phase_seconds(timings, "eval_a", "eval_b")
    stat, stat_path = newest_selection_stat(args.b_data_dir)

    delta = None
    if a and b:
        delta = b.get("accuracy", 0.0) - a.get("accuracy", 0.0)

    timing_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(r['phase'])}</td>"
        f"<td>{html.escape(r['status'])}</td>"
        f"<td>{html.escape(r['start'])}</td>"
        f"<td>{html.escape(r['end'])}</td>"
        f"<td>{fmt_seconds(r['seconds'])}</td>"
        f"<td><code>{html.escape(r['log'])}</code></td>"
        "</tr>"
        for r in timings
    )
    if not timing_rows:
        timing_rows = "<tr><td colspan='6'>No timing rows found.</td></tr>"

    stat_html = "<p>No Arm B selection stat file found yet.</p>"
    if stat:
        stat_html = (
            f"<p>Selection stat: <code>{html.escape(stat_path or '')}</code></p>"
            "<table><tbody>"
            f"<tr><th>Selected rows</th><td>{html.escape(str(stat.get('num_selected', 'n/a')))}</td></tr>"
            f"<tr><th>Data source</th><td><code>{html.escape(json.dumps(stat.get('data_source', {}), sort_keys=True))}</code></td></tr>"
            f"<tr><th>Pass-rate buckets</th><td><code>{html.escape(json.dumps(stat.get('pass_rate', {}), sort_keys=True))}</code></td></tr>"
            f"<tr><th>Diff-score buckets</th><td><code>{html.escape(json.dumps(stat.get('diff_score', {}), sort_keys=True))}</code></td></tr>"
            "</tbody></table>"
        )

    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    css = """
body{font-family:Inter,Arial,sans-serif;line-height:1.45;margin:32px;color:#1f2937;background:#f8fafc}
h1,h2{color:#111827}
.card{background:white;border:1px solid #d1d5db;border-radius:8px;padding:18px;margin:16px 0}
table{border-collapse:collapse;width:100%;background:white}
th,td{border:1px solid #d1d5db;padding:8px;text-align:left;vertical-align:top}
th{background:#eef2f7}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.92em}
.good{color:#047857}.bad{color:#b91c1c}.muted{color:#6b7280}
"""
    delta_class = "good" if (delta is not None and delta >= 0) else "bad"
    delta_text = "n/a" if delta is None else f"{delta * 100:+.2f} pp"
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>GSM8K Arm A vs Arm B Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>GSM8K Arm A vs Arm B Report</h1>
  <p class="muted">Run ID: <code>{html.escape(args.run_id)}</code> | Generated: {html.escape(generated)}</p>

  <div class="card">
    <h2>Summary</h2>
    <table>
      <tbody>
        <tr><th>Arm A</th><td>Full-data GRPO, Qwen2.5-3B, step 150</td></tr>
        <tr><th>Arm B</th><td>CROPI 10% selection, single round, Qwen2.5-3B, step 150</td></tr>
        <tr><th>Eval parquet</th><td><code>{html.escape(args.eval_parquet)}</code></td></tr>
        <tr><th>Accuracy delta (B - A)</th><td class="{delta_class}">{delta_text}</td></tr>
        <tr><th>A training time</th><td>{fmt_seconds(a_train_est)} <span class="muted">({html.escape(a_train_note)})</span></td></tr>
        <tr><th>B total pipeline time</th><td>{fmt_seconds(b_pipeline)}</td></tr>
        <tr><th>B score/select/train phase</th><td>{fmt_seconds(b_train)}</td></tr>
        <tr><th>Eval total time</th><td>{fmt_seconds(eval_total)}</td></tr>
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Accuracy</h2>
    <table>
      <thead><tr><th>Arm</th><th>Tag</th><th>Accuracy</th><th>Correct</th><th>Model</th></tr></thead>
      <tbody>
        {result_row("Arm A full data", a)}
        {result_row("Arm B CROPI 10%", b)}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Arm B Selection</h2>
    {stat_html}
  </div>

  <div class="card">
    <h2>Timings</h2>
    <table>
      <thead><tr><th>Phase</th><th>Status</th><th>Start</th><th>End</th><th>Elapsed</th><th>Log</th></tr></thead>
      <tbody>{timing_rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Artifacts</h2>
    <table>
      <tbody>
        <tr><th>Repo</th><td><code>{html.escape(args.repo)}</code></td></tr>
        <tr><th>A result</th><td><code>{html.escape(args.a_result)}</code></td></tr>
        <tr><th>B result</th><td><code>{html.escape(args.b_result)}</code></td></tr>
        <tr><th>A checkpoint root</th><td><code>{html.escape(args.a_ckpt_root)}</code></td></tr>
        <tr><th>B checkpoint root</th><td><code>{html.escape(args.b_ckpt_root)}</code></td></tr>
        <tr><th>Timings TSV</th><td><code>{html.escape(args.timings)}</code></td></tr>
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
