"""faster-whisper 系のモデルで文字起こしし、parakeet と同じ形式で出力する。

whisper_mcp の本番設定（幻聴対策パラメータ・beam_size）を再現したうえで、
VAD の有無を切り替えて比較できるようにする。

CTranslate2 形式であれば Whisper 派生モデルも同じ経路で測れる。
`--config` でモデルを選び、出力ファイル名の識別子（tag）もそれに従う。

使い方:
    uv run python scripts/run_whisper.py <音声ファイル> [--config whisper|turbo|kotoba]
                                         [--runs N] [--vad] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# tag -> モデル識別子。tag はそのまま出力ファイル名の構成要素になるので、
# 既存の結果ファイル（*__whisper__run1.json）と互換を保つため large-v3 は
# "whisper" のままにする。
MODELS = {
    "whisper": "large-v3",
    "turbo": "large-v3-turbo",
    "kotoba": "kotoba-tech/kotoba-whisper-v2.0-faster",
}

CACHE_DIR = r"E:\ModelCache\faster-whisper"

# whisper_mcp/src/whisper_mcp/engine.py と同じ値
HALLUCINATION_PARAMS = {
    "condition_on_previous_text": False,
    "no_speech_threshold": 0.6,
    "log_prob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_parakeet import to_srt  # noqa: E402  同じ srt 整形を使う


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--config", default="whisper", choices=sorted(MODELS))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--vad", action="store_true", help="VAD フィルタを有効にする")
    parser.add_argument("--outdir", default="data/outputs")
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.audio))[0]
    model_id = MODELS[args.config]
    tag = f"{args.config}-vad" if args.vad else args.config

    print(f"モデル読み込み: {model_id} (vad_filter={args.vad})")
    model = WhisperModel(model_id, device="cuda", compute_type="float16", download_root=CACHE_DIR)

    for run in range(1, args.runs + 1):
        print(f"--- run {run}/{args.runs} ---")
        # 速度比較のため、モデル読み込みを含まない転写だけの時間を測る。
        # transcribe は遅延評価なので、セグメントを消費しきるまでが1回分になる。
        started = time.perf_counter()
        segments_iter, info = model.transcribe(
            args.audio,
            language="ja",
            beam_size=5,
            vad_filter=args.vad,
            word_timestamps=True,  # 区間ごとの切り出し評価に使う
            **HALLUCINATION_PARAMS,
        )
        segments, units = [], []
        for s in segments_iter:
            segments.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip()})
            for w in s.words or []:
                units.append(
                    {"start": float(w.start), "end": float(w.end), "text": w.word.strip()}
                )
        elapsed = time.perf_counter() - started
        text = "".join(s["text"] for s in segments)

        base = os.path.join(args.outdir, f"{stem}__{tag}__run{run}")
        with open(f"{base}.txt", "w", encoding="utf-8") as fh:
            fh.write(text)
        with open(f"{base}.srt", "w", encoding="utf-8") as fh:
            fh.write(to_srt(segments))
        with open(f"{base}.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "text": text,
                    "segments": segments,
                    "units": units,
                    "model": model_id,
                    "elapsed_sec": round(elapsed, 3),
                    "audio_sec": round(float(info.duration), 3),
                },
                fh, ensure_ascii=False, indent=1,
            )

        rtf = elapsed / float(info.duration) if info.duration else float("nan")
        print(
            f"  セグメント数: {len(segments)} / 文字数: {len(text)}"
            f" / 転写 {elapsed:.2f}s（音声 {info.duration:.1f}s・実時間比 {rtf:.3f}）"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
