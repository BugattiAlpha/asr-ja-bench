"""日本語ASR出力の反復ループ・表記ゆれを測る。

注意: 日本語の反復を数えるときは必ず normalize_kana を通すこと。
ひらがな「ちょ」だけを数えてカタカナ「チョ」を落とすと、同じ現象が別々に
カウントされ、モデル間比較が無効になる（2026-07-19 に実際にやらかした）。
"""

from __future__ import annotations

import re
import unicodedata

_KATAKANA_START, _KATAKANA_END = ord("ァ"), ord("ヶ")
_KANA_OFFSET = ord("ァ") - ord("ぁ")


def normalize_kana(text: str) -> str:
    """カタカナ→ひらがな、全角半角、空白を畳んで比較可能な形にする。"""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        chr(ord(ch) - _KANA_OFFSET) if _KATAKANA_START <= ord(ch) <= _KATAKANA_END else ch
        for ch in text
    )
    return re.sub(r"\s+", "", text)


def longest_repeat_run(text: str, unit: str) -> int:
    """unit が連続して何回繰り返されたか、その最大値。表記ゆれは吸収する。"""
    norm_text = normalize_kana(text)
    norm_unit = normalize_kana(unit)
    if not norm_unit:
        return 0
    runs = re.findall(f"(?:{re.escape(norm_unit)})+", norm_text)
    return max((len(r) // len(norm_unit) for r in runs), default=0)


def repetition_score(text: str) -> float:
    """0.0〜1.0。1に近いほど同じ並びの繰り返しで埋まっている（縮退している）。

    文字2-gram のユニーク率の裏返し。反復ループは少数の2-gramで埋まるので高値になる。
    """
    norm = normalize_kana(text)
    if len(norm) < 2:
        return 0.0
    bigrams = [norm[i : i + 2] for i in range(len(norm) - 1)]
    return 1.0 - len(set(bigrams)) / len(bigrams)


def detect_window_loops(
    segments: list[dict], window: float = 30.0, tolerance: float = 0.05
) -> list[dict]:
    """処理ウィンドウ長ちょうどを1セグメントが占めている箇所を返す。

    Whisper 系のロングフォーム復号では、非音声区間などで30秒ウィンドウ全体が
    同一トークンの繰り返しで埋まり、その区間の発話が丸ごと失われる。
    「単一セグメントの長さ == ウィンドウ長」はその機械的な署名。

    実コンテンツの反復（ネタの掛け声など）は短いセグメントに分かれるため、
    この判定には引っかからない。
    """
    loops = []
    for seg in segments:
        duration = float(seg["end"]) - float(seg["start"])
        if duration >= window - tolerance:
            loops.append(
                {
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "duration": round(duration, 3),
                    "repetition_score": round(repetition_score(seg["text"]), 4),
                    "text": seg["text"][:120],
                }
            )
    return loops
