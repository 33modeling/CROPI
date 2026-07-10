"""full vs CROPI 두 arm의 요약 JSON을 읽어 한글 상세 리포트(Markdown)를 만든다.

eval_math.py가 남긴 <results_dir>/<dataset>_full.json 과 <dataset>_cropi10.json 을
읽어서, 정확도 비교뿐 아니라 응답 길이와 두 방식이 서로 다르게 푼 문제까지 정리해
<out>(기본값 <results_dir>/<dataset>_report.md)에 저장한다. 콘솔에는 핵심 숫자만
짧게 찍는다. 두 파일 중 하나가 없으면 있는 쪽만으로 리포트를 만든다.

    $RL_PYTHON cropi/eval/make_report.py \
        --results-dir $RESULTS_DIR --dataset gsm8k \
        --steps 60 --rounds 3 --select-ratio 10 \
        --out $RESULTS_DIR/gsm8k_report.md
"""
import argparse
import json
import os
from datetime import datetime

FULL_LABEL = "전체 데이터"
CROPI_LABEL = "CROPI"


def parse_args():
    ap = argparse.ArgumentParser(description="full vs CROPI 한글 리포트 생성")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--full-tag", default=None, help="기본값 <dataset>_full")
    ap.add_argument("--cropi-tag", default=None, help="기본값 <dataset>_cropi10")
    ap.add_argument("--steps", default=None, help="맞춰 놓은 학습 스텝 예산")
    ap.add_argument("--rounds", default=None, help="CROPI 라운드 수")
    ap.add_argument("--select-ratio", default="10", help="CROPI 선택 비율(%)")
    ap.add_argument("--samples", type=int, default=4, help="서로 다르게 푼 사례를 몇 개까지 실을지")
    ap.add_argument("--out", default=None, help="기본값 <results_dir>/<dataset>_report.md")
    return ap.parse_args()


def load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pct(x):
    return f"{x * 100:.2f}%"


def clip(text, n=110):
    return " ".join(str(text or "").split())[:n]


def align(full, cropi):
    """idx 기준으로 두 arm을 맞춘다 (같은 test parquet를 같은 순서로 평가하므로 정렬됨)."""
    fe = {e["idx"]: e for e in full.get("examples", [])}
    ce = {e["idx"]: e for e in cropi.get("examples", [])}
    common = sorted(set(fe) & set(ce))
    tally = {"both_ok": 0, "only_full": 0, "only_cropi": 0, "both_bad": 0}
    cropi_wins, full_wins = [], []
    for i in common:
        f_ok, c_ok = fe[i]["correct"], ce[i]["correct"]
        if f_ok and c_ok:
            tally["both_ok"] += 1
        elif f_ok and not c_ok:
            tally["only_full"] += 1
            full_wins.append((fe[i], ce[i]))
        elif c_ok and not f_ok:
            tally["only_cropi"] += 1
            cropi_wins.append((fe[i], ce[i]))
        else:
            tally["both_bad"] += 1
    return len(common), tally, cropi_wins, full_wins


def cropi_desc(ratio, rounds, steps):
    ratio = str(ratio).rstrip("%")
    parts = [f"매 라운드 상위 {ratio}%를 골라 학습"]
    if rounds not in (None, "", "None"):
        parts.append(f"{rounds}개 라운드로 나눠 진행")
    if steps not in (None, "", "None"):
        parts.append(f"라운드 합계 {steps} 스텝")
    return ", ".join(parts)


def render(full, cropi, args):
    L = []
    ds = args.dataset
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    ratio = str(args.select_ratio).rstrip("%")

    L.append(f"# {ds.upper()} 실험 결과 정리")
    L.append("")
    L.append(f"작성 시각 {now}. eval 단계가 남긴 요약 JSON을 바탕으로 두 학습 방식을 비교한 기록이다.")
    L.append("")

    # ── 요약 ────────────────────────────────────────────────────────────────
    L.append("## 요약")
    L.append("")
    if full and cropi:
        af, ac = full["accuracy"], cropi["accuracy"]
        delta = ac - af
        n = full.get("n", cropi.get("n", 0))
        head = (f"같은 학습량"
                + (f"({args.steps} 스텝)" if args.steps not in (None, "", "None") else "")
                + f"을 두고 데이터를 전부 쓴 방식과 CROPI로 상위 {ratio}%만 추린 방식을 "
                + f"{ds} 테스트 {n}문항에서 맞대어 봤다.")
        L.append(head)
        L.append("")
        L.append(f"{FULL_LABEL} 방식이 {pct(af)}, {CROPI_LABEL} 방식이 {pct(ac)}를 맞혔다.")
        if abs(delta) < 0.0005:
            L.append("두 방식의 정확도는 사실상 같은 수준으로, 어느 쪽이 낫다고 보기 어렵다.")
        else:
            winner = CROPI_LABEL if delta > 0 else FULL_LABEL
            rel = (abs(delta) / af * 100) if af > 0 else 0.0
            tail = f" (상대적으로 약 {rel:.1f}% 차이)" if af > 0 else ""
            L.append(f"{winner} 쪽이 {abs(delta) * 100:.2f}%p 앞섰다{tail}. "
                     + ("데이터의 10분의 1만 쓰고도 손해가 없었다는 뜻이다."
                        if winner == CROPI_LABEL else
                        "이번 설정에서는 선택이 오히려 손해였다."))
    elif full or cropi:
        one = full or cropi
        which = FULL_LABEL if full else CROPI_LABEL
        L.append(f"{which} 결과만 확보돼 단독으로 정리한다. "
                 f"{ds} 테스트 {one.get('n', 0)}문항에서 정확도는 {pct(one['accuracy'])}였다. "
                 "나머지 한쪽 결과가 채워지면 두 방식을 나란히 비교할 수 있다.")
    else:
        L.append("비교할 결과 파일을 찾지 못했다. eval 단계를 먼저 돌려야 한다.")
        return "\n".join(L) + "\n"
    L.append("")

    # ── 실험 구성 ─────────────────────────────────────────────────────────────
    ref = full or cropi
    L.append("## 실험 구성")
    L.append("")
    dec = ref.get("decode", {})
    temp = dec.get("temperature", 0.0)
    decode_line = ("greedy 디코딩" if temp in (0, 0.0) else f"temperature {temp} 디코딩")
    L.append(f"- 테스트셋: {ds} test, {ref.get('n', 0)}문항")
    L.append(f"- 채점: 학습과 같은 보상 함수(`{ref.get('reward', '?')}`)로 정답이면 1점")
    L.append(f"- 디코딩: 문항당 1개 응답, {decode_line}, 최대 {dec.get('max_tokens', '?')}토큰")
    L.append(f"- {FULL_LABEL}: 전체 학습 풀로 GRPO"
             + (f", {args.steps} 스텝" if args.steps not in (None, "", "None") else ""))
    L.append(f"- {CROPI_LABEL}: {cropi_desc(ratio, args.rounds, args.steps)}")
    L.append("")

    # ── 정확도 비교 ───────────────────────────────────────────────────────────
    L.append("## 정확도")
    L.append("")
    L.append("| 방식 | 정확도 | 정답/전체 |")
    L.append("| --- | --- | --- |")
    if full:
        L.append(f"| {FULL_LABEL} | {pct(full['accuracy'])} | {full['correct']}/{full['n']} |")
    if cropi:
        L.append(f"| {CROPI_LABEL} | {pct(cropi['accuracy'])} | {cropi['correct']}/{cropi['n']} |")
    L.append("")

    if not (full and cropi):
        L.append("_한쪽 결과만 있어 길이·오답 비교는 생략한다._")
        L.append("")
        return "\n".join(L) + "\n"

    # ── 응답 길이 ─────────────────────────────────────────────────────────────
    lf, lc = full.get("length", {}), cropi.get("length", {})
    L.append("## 응답 길이")
    L.append("")
    L.append(f"평균 응답 길이는 {FULL_LABEL}가 {lf.get('mean_tokens', '?')}토큰, "
             f"{CROPI_LABEL}가 {lc.get('mean_tokens', '?')}토큰이었다. "
             f"길이 상한에 걸려 잘린 응답은 각각 {lf.get('truncated', 0)}개, {lc.get('truncated', 0)}개다.")
    tf, tc = lf.get("truncated", 0), lc.get("truncated", 0)
    if max(tf, tc) > 0:
        L.append("잘린 응답은 답을 끝맺지 못해 오답으로 처리됐을 가능성이 높으니, "
                 "정확도를 볼 때 이 숫자를 같이 감안하는 편이 좋다.")
    L.append("")

    # ── 갈린 문제 ─────────────────────────────────────────────────────────────
    common, tally, cropi_wins, full_wins = align(full, cropi)
    L.append("## 두 방식이 갈린 문제")
    L.append("")
    if common == 0:
        L.append("예제별 기록이 정렬되지 않아(문항 수가 다르거나 예제 정보 없음) 문항 단위 비교는 건너뛴다.")
        L.append("")
        return "\n".join(L) + "\n"

    L.append(f"idx로 맞춰본 공통 {common}문항 기준이다.")
    L.append("")
    L.append(f"- 둘 다 맞힘: {tally['both_ok']}")
    L.append(f"- {FULL_LABEL}만 맞힘: {tally['only_full']}")
    L.append(f"- {CROPI_LABEL}만 맞힘: {tally['only_cropi']}")
    L.append(f"- 둘 다 틀림: {tally['both_bad']}")
    L.append("")
    swing = tally["only_cropi"] - tally["only_full"]
    if swing != 0:
        gainer = CROPI_LABEL if swing > 0 else FULL_LABEL
        L.append(f"한쪽만 맞힌 문항을 상쇄하면 {gainer}가 {abs(swing)}문항만큼 순이득을 봤다. "
                 "전체 정확도 차이도 결국 여기서 나온다.")
    else:
        L.append("한쪽만 맞힌 문항 수가 서로 비슷해, 정확도 차이가 특정 유형에 쏠려 있지는 않다.")
    L.append("")

    def dump_cases(title, cases):
        L.append(f"### {title}")
        L.append("")
        if not cases:
            L.append("해당하는 문항이 없다.")
            L.append("")
            return
        for fe, ce in cases[: args.samples]:
            q = clip(fe.get("question") or ce.get("question"))
            L.append(f"- {q}")
            L.append(f"  - 정답 `{fe['gold']}` · {FULL_LABEL} `{fe['pred'] or '못 찾음'}` · "
                     f"{CROPI_LABEL} `{ce['pred'] or '못 찾음'}`")
        if len(cases) > args.samples:
            L.append(f"- … 외 {len(cases) - args.samples}문항")
        L.append("")

    dump_cases(f"{CROPI_LABEL}만 맞힌 사례", cropi_wins)
    dump_cases(f"{FULL_LABEL}만 맞힌 사례", full_wins)

    # ── 메모 ─────────────────────────────────────────────────────────────────
    L.append("## 남겨두는 메모")
    L.append("")
    L.append("- 문항당 응답 하나(greedy)만 뽑아 채점한 값이라, 표본 분산은 반영돼 있지 않다.")
    L.append("- 사례에 적은 답은 `\\boxed{}` 등에서 뽑아낸 표시용이고, 맞고 틀림 자체는 보상 함수가 정한다.")
    L.append("- 두 방식은 같은 테스트 문항을 같은 순서로 풀었으므로 idx로 바로 짝지을 수 있다.")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    args = parse_args()
    full_tag = args.full_tag or f"{args.dataset}_full"
    cropi_tag = args.cropi_tag or f"{args.dataset}_cropi10"
    full = load(os.path.join(args.results_dir, full_tag + ".json"))
    cropi = load(os.path.join(args.results_dir, cropi_tag + ".json"))
    out = args.out or os.path.join(args.results_dir, f"{args.dataset}_report.md")

    md = render(full, cropi, args)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)

    # 콘솔에는 핵심만
    print(f"\n=== {args.dataset}: {FULL_LABEL} vs {CROPI_LABEL} ===")
    for label, r in ((FULL_LABEL, full), (CROPI_LABEL, cropi)):
        if r:
            print(f"  {label:10s} acc={r['accuracy']:.4f}  ({r['correct']}/{r['n']})")
        else:
            print(f"  {label:10s} (결과 없음)")
    if full and cropi:
        print(f"  차이 ({CROPI_LABEL}-{FULL_LABEL}) = {(cropi['accuracy'] - full['accuracy']) * 100:+.2f}%p")
    print(f"  리포트 -> {out}")


if __name__ == "__main__":
    main()
