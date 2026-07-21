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

# 実験ノート「その5」で公表した値（長音符バグ修正後）
PUBLISHED = {
    "broadcastmic": {"parakeet": 0.292, "whisper": 0.214},
    "chatmic": {"parakeet": 0.198, "whisper": 0.171},
    "v2_chatmic": {"parakeet": 0.089, "whisper": 0.093},
    "v2_broadcast": {"parakeet": 0.467, "whisper": 0.346},
    "ai_chatmic": {"parakeet": 0.066, "whisper": 0.055},
    "ai_broadcast": {"parakeet": 0.107, "whisper": 0.058},
}

RESULTS = os.path.join(ROOT, "data", "mictest_results")


def _load(name: str, config: str):
    paths = sorted(glob.glob(os.path.join(RESULTS, f"{name}__{config}__run*.json")))
    if not paths:
        return None
    with open(paths[0], encoding="utf-8") as fh:
        return json.load(fh)


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

    run = _load(name, config)
    if run is None:
        pytest.skip(f"{name}__{config} の結果が無い環境")
    got = report_real.score(references[name], run["text"])["cer"]
    assert round(got, 3) == pytest.approx(expected, abs=0.001), (
        f"{name}/{config}: 公開値 {expected} に対し {got:.4f}"
    )


def test_mictest_reference_is_284_chars():
    """参照の長さが変わったら、公開した CER の分母が変わっている。"""
    import report_real

    try:
        reference = report_real.mictest_reference()
    except FileNotFoundError:
        pytest.skip("読み上げ原稿 script.json が無い環境")
    assert len(reference) == 284
