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

# ⚠️ kotoba-whisper v2.0 で word_timestamps=True にすると、CTranslate2 が
#    アクセス違反（0xC0000005）でプロセスごと落ちる。例外ではなくクラッシュなので
#    try/except では拾えない。蒸留でデコーダが2層しかなく、単語アライメントが
#    参照する alignment heads と噛み合わないため（実測で切り分け済み: False なら正常）。
#    全文CERの算出には単語タイムスタンプを使わないので、この構成では切って測る。
#    区間ごとの切り出し評価（compare_mictest.py）はセグメント境界にフォールバックする。
WORD_TIMESTAMPS = {"kotoba": False}

# 分割窓の長さ（秒）。既定は 30（Whisper 本来の窓）。
# kotoba-whisper v2.0 のモデルカードが載せている faster-whisper 用の例は
# chunk_length=15 で、実測でもここが効く（ai_chatmic の CER 0.248 → 0.157）。
# 蒸留モデルは 30 秒窓での逐次デコードに弱く、既定のままだと不当に悪く出る。
CHUNK_LENGTH = {"kotoba": 15}

# ⚠️ chunk_length を transcribe に渡すと model.feature_extractor.n_samples を
#    **永続的に**書き換える。同じモデルオブジェクトで設定を切り替えて比較すると、
#    指定しなかった呼び出しが直前の値を引き継ぐ（実測: 30→15→未指定 で 15 のまま）。
#    このスクリプトは1プロセス1設定なので問題にならないが、対話的に比較するときは
#    モデルを作り直すこと。

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
    parser.add_argument("audio", nargs="+", help="音声ファイル（複数可・モデルは1回だけ読む）")
    parser.add_argument("--config", default="whisper", choices=sorted(MODELS))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--run-offset", type=int, default=0,
        help="出力の run 番号の開始位置-1。プロセスを分けて計測するときに使う",
    )
    parser.add_argument("--vad", action="store_true", help="VAD フィルタを有効にする")
    parser.add_argument("--outdir", default="data/outputs")
    parser.add_argument(
        "--chunk-length", type=int, default=None,
        help="分割窓の秒数。既定はモデルごとの推奨値（kotoba は 15、他は 30）",
    )
    parser.add_argument(
        "--tag", default=None,
        help="出力ファイル名の識別子。既定は --config の値（設定違いを別名で残すときに使う）",
    )
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    os.makedirs(args.outdir, exist_ok=True)
    model_id = MODELS[args.config]
    base_tag = args.tag or args.config
    tag = f"{base_tag}-vad" if args.vad else base_tag

    chunk_length = (
        args.chunk_length if args.chunk_length is not None
        else CHUNK_LENGTH.get(args.config)
    )

    print(f"モデル読み込み: {model_id} (vad_filter={args.vad}, chunk_length={chunk_length or 30})")
    model = WhisperModel(model_id, device="cuda", compute_type="float16", download_root=CACHE_DIR)

    word_timestamps = WORD_TIMESTAMPS.get(args.config, True)
    if not word_timestamps:
        print(f"  （{args.config} は単語タイムスタンプ非対応のため無効化して実行する）")

    for audio in args.audio:
        stem = os.path.splitext(os.path.basename(audio))[0]
        print(f"=== {stem} ===")
        transcribe_runs(
            model, audio, stem, tag, model_id, word_timestamps, chunk_length, args
        )
    return 0


def transcribe_runs(
    model, audio: str, stem: str, tag: str, model_id: str,
    word_timestamps: bool, chunk_length: int | None, args,
) -> None:
    for i in range(1, args.runs + 1):
        run = i + args.run_offset
        print(f"--- run {run} ---")
        # 速度比較のため、モデル読み込みを含まない転写だけの時間を測る。
        # transcribe は遅延評価なので、セグメントを消費しきるまでが1回分になる。
        extra = {"chunk_length": chunk_length} if chunk_length else {}
        started = time.perf_counter()
        segments_iter, info = model.transcribe(
            audio,
            language="ja",
            beam_size=5,
            vad_filter=args.vad,
            word_timestamps=word_timestamps,  # 区間ごとの切り出し評価に使う
            **HALLUCINATION_PARAMS,
            **extra,
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
                    "chunk_length": chunk_length or 30,
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


if __name__ == "__main__":
    sys.exit(main())
