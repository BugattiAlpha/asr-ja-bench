"""テストセットの結果を集計して比較表を出す。

各 (音声 × モデル構成) について:
  CER        : 読み正規化後の文字誤り率（低いほど良い）
  実行間一致 : 3回の出力が完全一致したか（デコードの安定性）
  反復スコア : 0-1。高いほど同じ並びで埋まっている（反復ループの疑い）
  ループ     : 処理ウィンドウ長ちょうどのセグメント数
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cer import cer  # noqa: E402
from metrics import detect_window_loops, longest_repeat_run, repetition_score  # noqa: E402

CONFIGS = ["parakeet", "whisper", "whisper-vad"]


def load_runs(results_dir: str, audio: str, config: str) -> list[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, f"{audio}__{config}__run*.json"))):
        with open(path, encoding="utf-8") as fh:
            runs.append(json.load(fh))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="data/testset")
    parser.add_argument("--results", default="data/results")
    parser.add_argument("--out", default="claudedocs/benchmark-results.json")
    args = parser.parse_args()

    with open(os.path.join(args.testset, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)

    report = {}
    print(f"{'音声':<12}{'モデル':<14}{'CER':>8}{'一致':>6}{'反復':>8}{'ループ':>7}{'文字数':>8}")
    print("-" * 65)

    for audio in manifest:
        ref_path = os.path.join(args.testset, f"{audio}.reference.txt")
        reference = open(ref_path, encoding="utf-8").read()
        report[audio] = {}

        for config in CONFIGS:
            runs = load_runs(args.results, audio, config)
            if not runs:
                continue
            texts = [r["text"] for r in runs]
            identical = len(set(texts)) == 1
            score = cer(reference, texts[0])
            rep = repetition_score(texts[0])
            loops = detect_window_loops(runs[0]["segments"])

            report[audio][config] = {
                "cer": round(score, 4),
                "runs_identical": identical,
                "distinct_outputs": len(set(texts)),
                "repetition_score": round(rep, 4),
                "window_loops": len(loops),
                "chars": len(texts[0]),
                "reference_chars": len(reference),
            }
            print(
                f"{audio:<12}{config:<14}{score:>8.3f}{'○' if identical else '×':>6}"
                f"{rep:>8.3f}{len(loops):>7}{len(texts[0]):>8}"
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f"\n保存: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
