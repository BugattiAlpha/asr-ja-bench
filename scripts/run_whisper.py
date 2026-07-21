"""faster-whisper large-v3 で文字起こしし、parakeet と同じ形式で出力する。

whisper_mcp の本番設定（幻聴対策パラメータ・beam_size）を再現したうえで、
VAD の有無を切り替えて比較できるようにする。

使い方:
    uv run python scripts/run_whisper.py <音声ファイル> [--runs N] [--vad] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

MODEL_SIZE = "large-v3"
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
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--vad", action="store_true", help="VAD フィルタを有効にする")
    parser.add_argument("--outdir", default="data/outputs")
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.audio))[0]
    tag = "whisper-vad" if args.vad else "whisper"

    print(f"モデル読み込み: {MODEL_SIZE} (vad_filter={args.vad})")
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16", download_root=CACHE_DIR)

    for run in range(1, args.runs + 1):
        print(f"--- run {run}/{args.runs} ---")
        segments_iter, _info = model.transcribe(
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
        text = "".join(s["text"] for s in segments)

        base = os.path.join(args.outdir, f"{stem}__{tag}__run{run}")
        with open(f"{base}.txt", "w", encoding="utf-8") as fh:
            fh.write(text)
        with open(f"{base}.srt", "w", encoding="utf-8") as fh:
            fh.write(to_srt(segments))
        with open(f"{base}.json", "w", encoding="utf-8") as fh:
            json.dump(
                {"text": text, "segments": segments, "units": units},
                fh, ensure_ascii=False, indent=1,
            )

        print(f"  セグメント数: {len(segments)} / 文字数: {len(text)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
