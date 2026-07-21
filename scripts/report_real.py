"""実発話6本の CER をモデル横断で1枚の表にする。

これまでマイクテスト4本とナレーション2本を別々に測っており、ナレーション側の
集計コードがリポジトリに残っていなかった。両者とも**全文どうしの CER** に統一する。

- マイクテスト4本: `mic_test_recorder` の読み上げ原稿（全11セクション連結・284字）が正解
- ナレーション2本: `data/mictest/ai_speech.reference.txt`（639字）が正解

区間ごとに切り出す評価は使わない。`section_bounds` は**予定のスケジュール**であって
実際の発話位置ではなく、話者が枠を越えて喋るとその分が欠落として計上されるため
（実験ノート「per-section CER は使えない」）。区間ごとの内訳や無音区間での捏造を
見たいときは `compare_mictest.py` を使う。

この方式で公開済みの parakeet / whisper の8値をすべて再現できることを確認済み
（回帰テスト: tests/test_report_real.py）。

CER の差は必ず文字数（編集距離）に換算して出す。参照長が違う音源どうしで
CER の差を直接比べると誤読するため、また 1文字差を「勝ち」と数えないため。

使い方:
    uv run python scripts/report_real.py --configs parakeet,whisper,turbo,kotoba
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cer import _levenshtein, to_reading  # noqa: E402
from compare_mictest import SCRIPT_JSON  # noqa: E402

# マイクテストの4本。読み上げ原稿が共通なので参照は同一。
# （音源と録音フォルダの対応は wav の MD5 で確認済み。v2_broadcast の実体は
#  NVIDIA-Broadcast のテイクで、同日の Ver2_Broadcast-Stream は使っていない）
MICTEST = ("broadcastmic", "chatmic", "v2_chatmic", "v2_broadcast")

NARRATION = ("ai_chatmic", "ai_broadcast")
NARRATION_REF = "data/mictest/ai_speech.reference.txt"

LABELS = {
    "broadcastmic": "マイクテスト Broadcast",
    "chatmic": "マイクテスト Chat-Mic",
    "v2_chatmic": "Ver2 Chat-Mic",
    "v2_broadcast": "Ver2 Broadcast",
    "ai_chatmic": "ナレーション Chat-Mic",
    "ai_broadcast": "ナレーション Broadcast",
}

ORDER = list(MICTEST) + list(NARRATION)


def load_runs(results_dir: str, name: str, config: str) -> list[dict]:
    runs = []
    for p in sorted(glob.glob(os.path.join(results_dir, f"{name}__{config}__run*.json"))):
        with open(p, encoding="utf-8") as fh:
            runs.append(json.load(fh))
    return runs


def mictest_reference() -> str:
    """読み上げ原稿の全セクションを連結した正解（284字）。

    「長音」セクション（あーーーー）は本来の正解を定義しにくいが、全モデルに
    同じだけ効くので除外しない。除外すると公開済みの数値と地続きでなくなる。
    """
    with open(SCRIPT_JSON, encoding="utf-8") as fh:
        script = json.load(fh)
    return "".join(s["text"].replace("\n", "") for s in script["sections"])


def score(reference: str, hypothesis: str) -> dict:
    """CER と、その内訳である編集距離・参照長（いずれも正規化後の文字数）。"""
    ref, hyp = to_reading(reference), to_reading(hypothesis)
    distance = _levenshtein(ref, hyp)
    return {
        "cer": distance / len(ref) if ref else 0.0,
        "errors": distance,
        "ref_chars": len(ref),
        "hyp_chars": len(hyp),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/mictest_results")
    parser.add_argument("--configs", default="parakeet,whisper")
    parser.add_argument("--out", default="claudedocs/real-recordings-report.json")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    with open(NARRATION_REF, encoding="utf-8") as fh:
        narration_ref = fh.read()
    references = {
        **{name: mictest_reference() for name in MICTEST},
        **{name: narration_ref for name in NARRATION},
    }

    report: dict[str, dict] = {}
    for name in ORDER:
        reference = references[name]

        row: dict[str, dict] = {}
        for config in configs:
            runs = load_runs(args.results, name, config)
            if not runs:
                continue
            entry = score(reference, runs[0]["text"])
            entry["runs"] = len(runs)
            entry["runs_identical"] = len({r["text"] for r in runs}) == 1
            elapsed = [r["elapsed_sec"] for r in runs if "elapsed_sec" in r]
            if elapsed:
                entry["elapsed_sec"] = round(statistics.median(elapsed), 3)
                audio = runs[0].get("audio_sec")
                if audio:
                    entry["rtf"] = round(entry["elapsed_sec"] / audio, 4)
            row[config] = entry
        if row:
            report[name] = row

    # --- 表 ---
    present = [c for c in configs if any(c in row for row in report.values())]
    head = "".join(f"{c[:10]:>11}" for c in present)
    print(f"\n{'録音':<24}{'参照字':>7}{head}    最良（誤り文字数）")
    print("-" * (34 + 11 * len(present) + 22))

    for name, row in report.items():
        ref_chars = next(iter(row.values()))["ref_chars"]
        cells = "".join(
            f"{row[c]['cer']:>11.3f}" if c in row else f"{'-':>11}" for c in present
        )
        scored = {c: row[c]["errors"] for c in present if c in row}
        fewest = min(scored.values())
        # 誤り2文字以内の差は勝敗として扱わない（1文字差を「勝ち」と数えない）
        best = [c for c, e in scored.items() if e <= fewest + 2]
        if len(scored) == 1:
            verdict = f"{best[0]}（単独・比較なし）"
        elif len(best) == len(scored):
            verdict = "差なし"
        else:
            verdict = "/".join(best)
        print(f"{LABELS[name]:<24}{ref_chars:>7}{cells}    {verdict}（{fewest}字）")

    # --- 誤り文字数（CERの分子そのもの） ---
    print(f"\n{'録音':<24}{'参照字':>7}" + "".join(f"{c[:10]:>11}" for c in present) + "   ← 誤り文字数")
    print("-" * (34 + 11 * len(present) + 18))
    for name, row in report.items():
        ref_chars = next(iter(row.values()))["ref_chars"]
        cells = "".join(
            f"{row[c]['errors']:>11}" if c in row else f"{'-':>11}" for c in present
        )
        print(f"{LABELS[name]:<24}{ref_chars:>7}{cells}")

    # --- 実行間の安定性 ---
    unstable = [
        (LABELS[name], c)
        for name, row in report.items()
        for c in present
        if c in row and not row[c]["runs_identical"]
    ]
    print("\n実行間で出力が揺れた組み合わせ: " + (
        "なし（全組で3回とも完全一致）" if not unstable
        else ", ".join(f"{n}/{c}" for n, c in unstable)
    ))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f"\n保存: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
