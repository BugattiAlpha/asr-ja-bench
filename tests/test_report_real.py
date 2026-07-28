"""公開済みの CER を再現できることを固定する回帰テスト。

集計方法を変えたときに、記事として世に出した数値と地続きでなくなることを防ぐ。
実測結果（data/mictest_results/）と録音側の原稿は公開リポジトリに含めていないため、
揃っていない環境では skip する。
"""

from __future__ import annotations

import glob
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# 公表した値。parakeet / whisper は実験ノート「その5」（長音符バグ修正後）、
# turbo / kotoba は続編記事（2026-07-21）の測定値。
PUBLISHED = {
    "broadcastmic": {"parakeet": 0.292, "whisper": 0.214, "turbo": 0.292, "kotoba": 0.342},
    "chatmic": {"parakeet": 0.198, "whisper": 0.171, "turbo": 0.128, "kotoba": 0.358},
    "v2_chatmic": {"parakeet": 0.089, "whisper": 0.093, "turbo": 0.125, "kotoba": 0.253},
    "v2_broadcast": {"parakeet": 0.467, "whisper": 0.346, "turbo": 0.315, "kotoba": 0.447},
    "ai_chatmic": {"parakeet": 0.066, "whisper": 0.055, "turbo": 0.058, "kotoba": 0.157},
    "ai_broadcast": {"parakeet": 0.107, "whisper": 0.058, "turbo": 0.057, "kotoba": 0.126},
}

RESULTS = os.path.join(ROOT, "data", "mictest_results")


def _load_runs(name: str, config: str):
    """その組み合わせの全 run を読む。

    run1 だけを見ると、実行ごとに出力が揺れる組み合わせ（実測では
    v2_broadcast×turbo）で「たまたま run1 がこの値だった」を公開値として
    固定してしまう。torch 2.6→2.13 の更新時に、精度は変わっていないのに
    このテストだけが落ちて発覚した。
    """
    paths = sorted(glob.glob(os.path.join(RESULTS, f"{name}__{config}__run*.json")))
    runs = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            runs.append(json.load(fh))
    return runs


@pytest.fixture(scope="module")
def references():
    import report_real

    try:
        mictest = report_real.mictest_reference()
    except FileNotFoundError:
        pytest.skip("読み上げ原稿 script.json が無い環境")
    narration_path = os.path.join(ROOT, report_real.NARRATION_REF)
    if not os.path.exists(narration_path):
        pytest.skip("ナレーションの参照テキストが無い環境")
    with open(narration_path, encoding="utf-8") as fh:
        narration = fh.read()
    return {
        **{n: mictest for n in report_real.MICTEST},
        **{n: narration for n in report_real.NARRATION},
    }


@pytest.mark.parametrize("name,config,expected", [
    (name, config, value)
    for name, row in PUBLISHED.items()
    for config, value in row.items()
])
def test_published_cer_is_reproduced(references, name, config, expected):
    import report_real

    runs = _load_runs(name, config)
    if not runs:
        pytest.skip(f"{name}__{config} の結果が無い環境")

    cers = [report_real.score(references[name], run["text"])["cer"] for run in runs]
    rounded = [round(c, 3) for c in cers]

    # 全 run が同じ出力なら、公開値と一致することを厳密に見る。
    if len({run["text"] for run in runs}) == 1:
        assert rounded[0] == pytest.approx(expected, abs=0.001), (
            f"{name}/{config}: 公開値 {expected} に対し {cers[0]:.4f}（全 run 同一）"
        )
        return

    # 出力が実行ごとに揺れる組み合わせは、公開値がその実測の幅に収まっていれば良しとする。
    # 幅から外れたら、揺らぎではなく本当に何かが変わっている。
    lo, hi = min(rounded), max(rounded)
    assert lo - 0.001 <= expected <= hi + 0.001, (
        f"{name}/{config}: 公開値 {expected} が実測の幅 {lo}〜{hi} の外にある"
        f"（各 run: {[f'{c:.4f}' for c in cers]}）"
    )


def test_mictest_reference_is_284_chars():
    """参照の長さが変わったら、公開した CER の分母が変わっている。"""
    import report_real

    try:
        reference = report_real.mictest_reference()
    except FileNotFoundError:
        pytest.skip("読み上げ原稿 script.json が無い環境")
    assert len(reference) == 284


def test_pins_actually_run_when_data_is_present():
    """実測データがあるのに全ピンが skip される事故を検出する。

    ピンの skip は「データが無い公開クローン」を通すためのもので、
    データが手元にあるならピンは実行されなければならない。skip だけで
    緑になると、この回帰テストは何も守っていないことになる。
    """
    if not glob.glob(os.path.join(RESULTS, "*__run1.json")):
        pytest.skip("実測データが無い環境（公開クローン）")
    loadable = sum(
        1
        for name, row in PUBLISHED.items()
        for config in row
        if _load_runs(name, config)
    )
    assert loadable > 0, "実測データがあるのに公開値のピンが1件も実行されていない"


def test_corrupted_bounds_are_rejected(tmp_path):
    """区間が1つ混入した録音を別レジームと誤読しないことを固定する。

    19区間の録音に発話級の区間を1つ足すと20区間になり、区間数だけの判定では
    手動プロンプター進行として「解釈できてしまう」。長さ検証が拒否することを確認する。
    """
    import json as _json

    import compare_mictest

    src = os.path.join(
        r"C:\Users\Cooliris\MY_Work_Space\mic_test_recorder\recordings",
        "20260720_194031_15cm_BroadcastMic",
        "meta.json",
    )
    if not os.path.exists(src):
        pytest.skip("録音メタが無い環境")
    with open(src, encoding="utf-8") as fh:
        meta = _json.load(fh)
    assert len(meta["section_bounds"]) == 19
    # 末尾に発話級（10秒）の区間を混入させて20区間にする
    last_end = float(meta["section_bounds"][-1][1])
    meta["section_bounds"].append([last_end, last_end + 10.0])
    (tmp_path / "meta.json").write_text(
        _json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        compare_mictest.load_sections(str(tmp_path))
