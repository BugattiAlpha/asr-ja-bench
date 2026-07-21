"""日本語向けの CER（文字誤り率）計算。

漢字/かなの表記差は誤りではないので、両者を読み（ひらがな）へ正規化してから
編集距離を取る。句読点・空白は ASR が出したり出さなかったりするため除去する。
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# ⚠️ 長音符「ー」を入れてはいけない。句読点と違って音韻的に弁別的で、
#    落とすと「ビール」と「ビル」が同一になり、ASRがよくやる長音の脱落が
#    誤りとして計上されなくなる（CERの過小評価）。
_PUNCT = re.compile(r"[、。，．,\.\s・「」『』（）\(\)!！?？…―~〜:：;；\"']")


@lru_cache(maxsize=1)
def _converter():
    import pykakasi

    return pykakasi.kakasi()


def to_reading(text: str) -> str:
    """漢字・カタカナ混じり文をひらがなの読みへ正規化する。"""
    text = unicodedata.normalize("NFKC", text)
    parts = _converter().convert(text)
    reading = "".join(p["hira"] for p in parts)
    # 大文字小文字の違いは書き起こしの誤りではない。parakeet は "ai"、
    # whisper は "AI" と出すため、畳まないと casing だけでモデル比較が歪む。
    return _PUNCT.sub("", reading).lower()


def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """読みへ正規化した上での文字誤り率。挿入が多いと 1.0 を超える。"""
    ref = to_reading(reference)
    hyp = to_reading(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)
