"""反復ループ検出・表記正規化の検証。

2026-07-19 に、ひらがな「ちょ」だけを数えてカタカナ「チョ」を見落とし、
異なる基準の数値を並べて「幻聴ループだ」と誤った結論を出した。
その再発を防ぐのがこのテストの目的。
"""

from metrics import (
    detect_window_loops,
    longest_repeat_run,
    normalize_kana,
    repetition_score,
)


def test_normalize_kana_unifies_katakana_and_hiragana():
    """カタカナ「チョ」とひらがな「ちょ」を同一視できること（過去の誤集計の直接の原因）。"""
    assert normalize_kana("チョチョン") == normalize_kana("ちょちょん")


def test_normalize_kana_folds_fullwidth_and_spacing():
    assert normalize_kana("ハイ　ハイ ハイ") == normalize_kana("はいはいはい")


def test_longest_repeat_run_counts_across_kana_forms():
    """表記が混在した反復も1本の連続として数えること。"""
    text = "ちょちょチョチョちょ"
    assert longest_repeat_run(text, unit="ちょ") == 5


def test_longest_repeat_run_ignores_non_adjacent_occurrences():
    """離れた出現は連続として数えない（実コンテンツの反復を誤検出しない）。"""
    text = "ちょちょん、こわいよ、ちょちょん"
    assert longest_repeat_run(text, unit="ちょ") == 2


def test_repetition_score_flags_degenerate_text():
    """同一トークンで埋まったテキストは高スコア、通常の発話は低スコア。"""
    loop = "ハイ " * 75
    normal = "どうもチャンピオンズです。今日は漫才をやります。よろしくお願いします。"
    assert repetition_score(loop) > 0.8
    assert repetition_score(normal) < 0.3


def test_detect_window_loops_flags_segment_filling_the_whole_window():
    """処理ウィンドウ長ちょうどを1セグメントが占めていたらループとして検出する。"""
    segments = [
        {"start": 0.0, "end": 2.5, "text": "どうも"},
        {"start": 124.78, "end": 154.78, "text": "ハイ " * 75},  # ちょうど30.000秒
        {"start": 159.38, "end": 160.78, "text": "チョチョンチョンチョチョン"},
    ]
    loops = detect_window_loops(segments, window=30.0)
    assert len(loops) == 1
    assert loops[0]["start"] == 124.78


def test_detect_window_loops_ignores_real_repeated_content():
    """実コンテンツの反復（短いセグメントに分かれる）はループ扱いしない。

    ターミネーターのテーマを「ちょ」で表現した箇所が該当。
    """
    segments = [
        {"start": 159.38, "end": 160.78, "text": "チョチョンチョンチョチョン"},
        {"start": 161.78, "end": 163.78, "text": "チョチョンチョンチョチョン"},
        {"start": 166.78, "end": 167.78, "text": "チョチョンチョンチョチョン"},
    ]
    assert detect_window_loops(segments, window=30.0) == []


def test_detect_window_loops_tolerates_float_jitter():
    """29.98秒のような端数でもウィンドウ長とみなす（タイムスタンプの丸め対策）。"""
    segments = [{"start": 0.0, "end": 29.98, "text": "あ " * 60}]
    assert len(detect_window_loops(segments, window=30.0, tolerance=0.05)) == 1
