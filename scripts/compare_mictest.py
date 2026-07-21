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


# 読み飛ばす無音区間の長さ上限（秒）。実測では gap が 2.00〜2.25s、
# 手動進行のクリック区間が 0.50s、発話セクションは最短でも 8.4s なので
# 5秒できれいに分離できる。
_MAX_GAP_SEC = 5.0


def _walk(script_sections: list[dict], bounds: list, has_gap) -> list[dict] | None:
    """has_gap(sec) が真のセクションの直後を無音区間として読み飛ばしつつ歩く。

    区間をちょうど使い切り、かつ読み飛ばした区間がすべて無音らしい長さ
    （_MAX_GAP_SEC 未満）のときだけ結果を返す。区間数だけで判定すると、
    余分な区間が1つ混入した 19+1 区間の録音を別レジームと誤読し、
    全セクションの座標が1つずれた「もっともらしい嘘」を返しうる。
    読み飛ばし対象に発話級の長さ（8秒超）が入っていたらレジーム違いとみなす。
    """
    sections, idx = [], 0
    for sec in script_sections:
        if idx >= len(bounds):
            return None
        start, end = bounds[idx]
        idx += 1
        if has_gap(sec):
            if idx < len(bounds):
                g_start, g_end = bounds[idx]
                if float(g_end) - float(g_start) >= _MAX_GAP_SEC:
                    return None  # 読み飛ばそうとした区間が無音の長さではない
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
    return sections if idx == len(bounds) else None


def load_sections(recording_dir: str) -> list[dict]:
    with open(SCRIPT_JSON, encoding="utf-8") as fh:
        script = json.load(fh)
    with open(os.path.join(recording_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)

    # section_bounds はセクション本体と無音を**交互に**含む。単純に zip すると
    # 全体がずれるので、無音を消費しながら歩く。
    # 無音の入り方は録音レジームで2通りあり、区間数が変わる:
    #   タイマー進行（script_version 1.0 / 19区間）
    #     … gap_after > 0 のセクションの直後だけ
    #   手動プロンプター進行（2.0 / 20区間）
    #     … 発話セクションすべての直後。gap_after=0 の「数字カウント」の後にも
    #       クリック操作ぶんの短い区間（実測 0.50s）が入る
    # どちらで歩けば区間を使い切るかで判定する（script_version は録音アプリ側の
    # 版であって、この境界構造を直接には示さないため、実データで決める）。
    bounds = meta["section_bounds"]
    for has_gap in (
        lambda sec: bool(sec.get("gap_after", 0)),
        lambda sec: sec["kind"] == "speech",
    ):
        sections = _walk(script["sections"], bounds, has_gap)
        if sections is not None:
            return sections
    raise ValueError(
        f"section_bounds({len(bounds)}区間) を既知のどちらのレジームでも解釈できない: "
        f"{recording_dir}"
    )


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
    parser.add_argument(
        "--configs",
        default="parakeet,whisper",
        help="比較するモデルの tag をカンマ区切りで（例: parakeet,whisper,turbo,kotoba）",
    )
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    sections = load_sections(args.recording_dir)
    lexical = [s for s in sections if s["kind"] == "speech" and s["id"] not in NON_LEXICAL_SECTIONS]
    silences = [s for s in sections if s["kind"] == "silence"]
    reference_full = "".join(s["text"] for s in lexical)

    print(f"\n{'='*78}\n{args.name}  （正解 {len(reference_full)}文字 / 評価対象 {len(lexical)}セクション）\n{'='*78}")

    runs_by_config = {}
    for config in configs:
        paths = sorted(glob.glob(os.path.join(args.results, f"{args.name}__{config}__run*.json")))
        runs = []
        for p in paths:
            with open(p, encoding="utf-8") as fh:
                runs.append(json.load(fh))
        # 文字/単語単位のタイムスタンプが無いモデル（例: kotoba は
        # word_timestamps 非対応）は、区間切り出しが 15〜30 秒のセグメント単位に
        # 落ちて他モデルと比較できない数字になる。黙って劣化させず除外する。
        # 全文どうしの CER は report_real.py で測れる。
        if runs and not runs[0].get("units"):
            print(f"  ⚠ {config}: 単語タイムスタンプが無いため区間評価から除外"
                  f"（全文CERは report_real.py を使う）")
            continue
        runs_by_config[config] = runs

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
    # ⚠️ section_bounds は「予定のスケジュール」であって実際の発話位置ではない。
    #    話者が枠を越えて喋ると欠落扱いになるため、per-section CER は
    #    モデルの優劣ではなく「どこで落としたか」を探す手がかりとしてのみ使う。
    active = [c for c in configs if runs_by_config.get(c)]
    header = "".join(f"{c[:9]:>10}" for c in active)
    print(f"\n{'セクション':<22}{header}   最良")
    print("-" * (25 + 10 * len(active) + 12))
    for s in lexical:
        row = {
            c: cer(s["text"], slice_hypothesis(runs_by_config[c][0], s["start"], s["end"]))
            for c in active
        }
        if not row:
            continue
        best_score = min(row.values())
        # 差が僅かなものを「勝ち」と呼ばない（1文字差を勝敗にしないため）
        winners = [c for c, v in row.items() if v <= best_score + 0.001]
        mark = "同等" if len(winners) == len(row) else "/".join(winners)
        cells = "".join(f"{row[c]:>10.3f}" for c in active)
        print(f"{s['label'][:20]:<22}{cells}   {mark}")
        summary.setdefault("per_section", {})[s["label"]] = {
            c: round(v, 4) for c, v in row.items()
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
