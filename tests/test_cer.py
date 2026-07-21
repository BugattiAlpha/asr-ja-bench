"""CER 計算と読み正規化の検証。

日本語ASRの CER は、漢字/かなの表記差をそのまま誤りに数えると不当に悪化する
（参照「きょうとし」に対しモデル出力「京都市」は正解なのに3文字誤りになる）。
読みへ正規化してから比較することを、このテストで固定する。
"""

import pytest
from cer import cer, to_reading


def test_to_reading_converts_kanji_to_hiragana():
    assert to_reading("京都市") == "きょうとし"


def test_to_reading_unifies_katakana_with_hiragana():
    assert to_reading("コンビ") == to_reading("こんび")


def test_to_reading_drops_punctuation_and_spaces():
    """ASR は句読点を出したり出さなかったりするので、比較対象から外す。"""
    assert to_reading("こんにちは、今日は。") == to_reading("こんにちは 今日は")


def test_cer_is_zero_for_reading_equivalent_text():
    """表記が違っても読みが同じなら誤り0。"""
    assert cer("きょうとし に いきます", "京都市に行きます") == 0.0


def test_cer_counts_substitution():
    # 「ねこ」→「いぬ」は2文字置換 / 参照は4文字（ねことけん…ではなく単純化）
    assert cer("ねこ", "いぬ") == 1.0


def test_cer_counts_deletion_as_error():
    """欠落（30秒まるごと消えるケース）が誤りとして効くこと。"""
    assert cer("あいうえお", "あい") == pytest.approx(0.6)


def test_cer_is_one_when_hypothesis_is_empty():
    assert cer("あいうえお", "") == 1.0


def test_cer_penalizes_repetition_loop_insertions():
    """反復ループによる大量挿入が誤りとして効くこと（CERは1を超えうる）。"""
    assert cer("はい", "はい" * 50) > 1.0


def test_to_reading_folds_letter_case():
    """大文字小文字の違いは書き起こしの誤りではない。

    parakeet は "ai"、whisper は "AI" と出力する。台本に AI が10回出ると、
    casing だけで CER が約3ポイント動き、モデル比較が壊れる。
    """
    assert to_reading("AIとMachine Learning") == to_reading("aiとmachine learning")


def test_cer_ignores_case_difference():
    assert cer("AIは道具である", "aiは道具である") == 0.0


def test_to_reading_keeps_long_vowel_mark():
    """長音符は句読点と違い音韻的に弁別的。落とすと実在の脱落誤りが計上されない。

    ASRが長音を落とすのはよくある誤りなので、「ビール」と「ビル」は
    別物として数える必要がある。
    """
    assert to_reading("ビール") != to_reading("ビル")


def test_cer_counts_dropped_long_vowel_as_error():
    assert cer("コーヒーを飲む", "コヒを飲む") > 0.0
