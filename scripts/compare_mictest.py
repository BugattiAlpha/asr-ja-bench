"""mic_test_recorder の録音を、読み上げ原稿を正解として評価する。

script.json に原稿、meta.json に各セクションの時刻境界があるので、
全体CERに加えて「どのセクションで落としたか」「無音区間で捏造したか」まで測れる。

使い方:
    uv run python scripts/compare_mictest.py <録音フォルダ> --results DIR --name NAME
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cer import cer, to_reading  # noqa: E402

SCRIPT_JSON = r"C:\Users\Cooliris\MY_Work_Space\mic_test_recorder\script.json"

# 長音「あーーーーー」や「「あ」…「あ」」は文章ではなく音の試験で、
# 書き起こしの正解を定義できない。全体CERからは外し、別枠で中身を見る。
NON_LEXICAL_SECTIONS = {6, 9}


def load_sections(recording_dir: str) -> list[dict]:
    with open(SCRIPT_JSON, encoding="utf-8") as fh:
        script = json.load(fh)
    with open(os.path.join(recording_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)

    # section_bounds はセクション本体と gap_after の無音を**交互に**含む
    # （11セクションに対し19区間）。単純に zip すると全体がずれるので、
    # gap_after を消費しながら歩く。
    bounds = meta["section_bounds"]
    sections = []
    idx = 0
    for sec in script["sections"]:
        start, end = bounds[idx]
        idx += 1
        if sec.get("gap_after", 0):
            idx += 1  # 直後の無音区間を読み飛ばす
        sections.append(
            {
                "id": sec["id"],
                "label": sec["label"],
                "kind": sec["kind"],
                "text": sec["text"].replace("\n", ""),
                "start": float(start),
                "end": float(end),
            }
        )
    if idx != len(bounds):
        raise ValueError(f"section_bounds の消費数が合わない: {idx} != {len(bounds)}")
    return sections


def slice_hypothesis(run: dict, start: float, end: float) -> str:
    """区間に入る文字/単語を連結する。

    セグメント単位で切ると、無音でまとめたセグメントが区間境界をまたいだときに
    隣のセクションの発話まで巻き込み、そのセクションのCERが不当に悪化する
    （実際 parakeet の 12.48-23.76 秒の1セグメントで起きた）。
    文字・単語単位のタイムスタンプがあればそれを使う。
    """
    items = run.get("units") or run["segments"]
    out = []
    for item in items:
        mid = (float(item["start"]) + float(item["end"])) / 2
        if start <= mid < end:
            out.append(item["text"])
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_dir")
    parser.add_argument("--results", default="data/mictest_results")
    parser.add_argument("--name", required=True, help="出力ファイルの stem")
    args = parser.parse_args()

    sections = load_sections(args.recording_dir)
    lexical = [s for s in sections if s["kind"] == "speech" and s["id"] not in NON_LEXICAL_SECTIONS]
    silences = [s for s in sections if s["kind"] == "silence"]
    reference_full = "".join(s["text"] for s in lexical)

    print(f"\n{'='*78}\n{args.name}  （正解 {len(reference_full)}文字 / 評価対象 {len(lexical)}セクション）\n{'='*78}")

    runs_by_config = {}
    for config in ["parakeet", "whisper"]:
        paths = sorted(glob.glob(os.path.join(args.results, f"{args.name}__{config}__run*.json")))
        runs_by_config[config] = [json.load(open(p, encoding="utf-8")) for p in paths]

    # --- 全体CER ---
    print(f"\n{'モデル':<12}{'全体CER':>10}{'実行間一致':>12}{'文字数':>8}")
    print("-" * 44)
    summary = {}
    for config, runs in runs_by_config.items():
        if not runs:
            continue
        hyp_lex = "".join(
            slice_hypothesis(runs[0], s["start"], s["end"]) for s in lexical
        )
        score = cer(reference_full, hyp_lex)
        identical = len({r["text"] for r in runs}) == 1
        summary[config] = {"overall_cer": round(score, 4), "runs_identical": identical}
        print(f"{config:<12}{score:>10.3f}{'○' if identical else '×':>12}{len(hyp_lex):>8}")

    # --- セクション別CER ---
    print(f"\n{'セクション':<22}{'parakeet':>10}{'whisper':>10}   優劣")
    print("-" * 60)
    for s in lexical:
        row = {}
        for config, runs in runs_by_config.items():
            if not runs:
                continue
            row[config] = cer(s["text"], slice_hypothesis(runs[0], s["start"], s["end"]))
        if len(row) < 2:
            continue
        p, w = row["parakeet"], row["whisper"]
        mark = "parakeet" if p < w - 0.001 else ("whisper" if w < p - 0.001 else "同等")
        print(f"{s['label'][:20]:<22}{p:>10.3f}{w:>10.3f}   {mark}")
        summary.setdefault("per_section", {})[s["label"]] = {
            "parakeet": round(p, 4), "whisper": round(w, 4)
        }

    # --- 無音区間の捏造 ---
    print(f"\n{'無音区間での捏造（正解は空）':<30}")
    print("-" * 60)
    for s in silences:
        for config, runs in runs_by_config.items():
            if not runs:
                continue
            text = slice_hypothesis(runs[0], s["start"], s["end"]).strip()
            state = f"「{text}」" if text else "（出力なし・正しい）"
            print(f"  {s['label']:<10}{config:<10}{state}")
            summary.setdefault("silence", {}).setdefault(s["label"], {})[config] = text

    # --- 非語彙セクションの中身 ---
    print(f"\n{'長音・間（正解を定義できないため参考）':<30}")
    print("-" * 60)
    for s in sections:
        if s["id"] not in NON_LEXICAL_SECTIONS:
            continue
        for config, runs in runs_by_config.items():
            if not runs:
                continue
            text = slice_hypothesis(runs[0], s["start"], s["end"]).strip()
            print(f"  {s['label'][:12]:<14}{config:<10}「{text[:50]}」")

    out = os.path.join("claudedocs", f"mictest-{args.name}.json")
    os.makedirs("claudedocs", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
