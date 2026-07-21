"""同一音源・同一マシンでの推論速度を並べる。

記事の速度比較が「0.6B の parakeet 対 1.55B の large-v3」という不公平な対比に
なっていたため、同じ Whisper 系列の中で large-v3 / large-v3-turbo /
Kotoba-Whisper を並べられるようにする。

⚠️ 各回は**別プロセス**で測ること（`--run-offset` で run 番号をずらす）。
   同じプロセスで transcribe を繰り返すと parakeet だけ時間が単調に伸び、
   実測で 5.7秒 → 11.0秒 と約2倍に膨らんだ（whisper 系は安定）。
   同一プロセスで5回まわして中央値を採ると、parakeet の速度を 85% 過大に報告する。

⚠️ 比較の前提（そろっていないもの）
- Whisper 系3つは完全に同条件（同じコード経路・beam_size=5・float16）。
- parakeet だけは NeMo で経路が違い、`elapsed_sec` に 16kHz 変換の時間を
  含まない（変換は計測の外で1回だけ行う）。Whisper 系は transcribe の中で
  音声を読むぶんが入る。parakeet 側がわずかに有利に出る。
- モデル読み込み時間は含めない（常駐運用を想定）。
- GPU を他のプロセスが使っていると絶対値は伸びる。比較は必ず同じ日・
  同じ状態で連続して測ったものどうしで行う。

使い方（各モデル3回・別プロセス）:
    uv run python scripts/run_whisper.py <音声> --config turbo --run-offset 0 --outdir data/speed_results
    uv run python scripts/run_whisper.py <音声> --config turbo --run-offset 1 --outdir data/speed_results
    ...
    uv run python scripts/report_speed.py --results data/speed_results
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

LABELS = {
    "parakeet": "parakeet-tdt_ctc-0.6b-ja",
    "whisper": "faster-whisper large-v3",
    "turbo": "faster-whisper large-v3-turbo",
    "kotoba": "kotoba-whisper-v2.0-faster",
}

PARAMS = {  # 公称パラメータ数（速度差の解釈用）
    "parakeet": "0.6B",
    "whisper": "1.55B",
    "turbo": "0.81B",
    "kotoba": "0.76B",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/speed_results")
    parser.add_argument("--audio", default="ai_chatmic", help="計測に使った音源の stem")
    parser.add_argument("--configs", default="parakeet,whisper,turbo,kotoba")
    parser.add_argument("--out", default="claudedocs/speed-report.json")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    report = {}

    for config in configs:
        runs = []
        for p in sorted(
            glob.glob(os.path.join(args.results, f"{args.audio}__{config}__run*.json"))
        ):
            with open(p, encoding="utf-8") as fh:
                runs.append(json.load(fh))
        times = [r["elapsed_sec"] for r in runs if "elapsed_sec" in r]
        if not times:
            continue
        audio_sec = runs[0]["audio_sec"]
        median = statistics.median(times)
        report[config] = {
            "model": runs[0].get("model"),
            "audio_sec": audio_sec,
            "runs": times,
            "median_sec": round(median, 3),
            "rtf": round(median / audio_sec, 4),
            "speedup_vs_realtime": round(audio_sec / median, 1),
            "chars": len(runs[0]["text"]),
        }

    if not report:
        print("計測結果が見つからない")
        return 1

    # 全構成が同じ音源を測っていることを確認する。違う長さの音源が混ざると、
    # 最遅比や見出しの音声長が意味を失う。
    durations = {c: v["audio_sec"] for c, v in report.items()}
    if max(durations.values()) - min(durations.values()) > 0.05:
        raise SystemExit(f"音源の長さが構成間で一致しない: {durations}")

    # 各構成の実測値の開き（最大/最小）を見る。1.25倍を超えていたら、
    # 計測中に GPU の混み具合が変わった疑いが強く、中央値が「速い状態」とも
    # 「遅い状態」とも言えない数字になる。この状態の絶対値・比率は公開しない。
    unstable = {
        c: round(max(v["runs"]) / min(v["runs"]), 2)
        for c, v in report.items()
        if min(v["runs"]) > 0 and max(v["runs"]) / min(v["runs"]) > 1.25
    }
    if unstable:
        print(f"\n⚠ 実測値の開きが大きい構成: {unstable}（GPU競合の疑い。測り直しを推奨）")

    audio_sec = next(iter(report.values()))["audio_sec"]
    slowest = max(v["median_sec"] for v in report.values())
    print(f"\n音源: {args.audio}（{audio_sec:.1f}秒）／中央値・モデル読み込みを含まない")
    print(f"\n{'モデル':<32}{'規模':>7}{'転写秒':>9}{'実時間比':>10}{'実時間の何倍':>13}{'最遅比':>9}")
    print("-" * 82)
    for config, v in sorted(report.items(), key=lambda kv: kv[1]["median_sec"]):
        print(
            f"{LABELS.get(config, config):<32}{PARAMS.get(config, '?'):>7}"
            f"{v['median_sec']:>9.2f}{v['rtf']:>10.3f}"
            f"{v['speedup_vs_realtime']:>12.1f}x{slowest / v['median_sec']:>8.1f}x"
        )

    print("\n各回の実測値（ばらつき確認用）")
    for config, v in report.items():
        print(f"  {config:<10}{', '.join(f'{t:.2f}s' for t in v['runs'])}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f"\n保存: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
