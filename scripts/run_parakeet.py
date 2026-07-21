"""nvidia/parakeet-tdt_ctc-0.6b-ja で文字起こしし、txt/srt/json を出力する。

使い方:
    uv run python scripts/run_parakeet.py <音声ファイル> [--runs N] [--outdir DIR]

NeMo は 16kHz モノラル wav を前提とするため、ffmpeg で変換してから渡す。
複数回実行するのは、デコードの実行間ばらつき（Whisper では実測で大きかった）を
測るため。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import wave

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

MODEL_ID = "nvidia/parakeet-tdt_ctc-0.6b-ja"


def to_wav16k(src: str) -> str:
    """16kHz モノラル wav へ変換したパスを返す。"""
    fd, dst = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = ["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", "-vn", dst]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 失敗 (rc={proc.returncode}): {proc.stderr[-800:]}")
    return dst


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(segments: list[dict]) -> str:
    out = []
    for i, seg in enumerate(segments, 1):
        out.append(
            f"{i}\n{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n{seg['text'].strip()}\n"
        )
    return "\n".join(out)


def group_words(words: list[dict], gap: float = 0.6) -> list[dict]:
    """単語タイムスタンプを無音の切れ目でまとめてセグメント化する。

    このモデルは句読点をほとんど出力しないため、NeMo の句読点ベースの
    セグメント分割では全体が1セグメントになってしまい、Whisper の srt と
    粒度が揃わない。無音 gap 秒以上で切ることで比較可能な単位にする。
    """
    segments: list[dict] = []
    for w in words:
        start, end = float(w["start"]), float(w["end"])
        text = str(w.get("word", w.get("segment", ""))).strip()
        if not text:
            continue
        if segments and start - segments[-1]["end"] < gap:
            segments[-1]["end"] = end
            segments[-1]["text"] += text
        else:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as fh:
        return fh.getnframes() / float(fh.getframerate())


def transcribe(model, wav: str) -> dict:
    """1回分の文字起こし結果（全文＋セグメント）を返す。

    elapsed_sec は whisper 側と揃えるため、モデル読み込みと 16kHz 変換を含まない
    推論そのものの時間にする。
    """
    started = time.perf_counter()
    results = model.transcribe([wav], timestamps=True)
    elapsed = time.perf_counter() - started
    hyp = results[0]

    stamps = getattr(hyp, "timestamp", None) or {}

    # 日本語は分かち書きしないため "word" は全文1語になる。文字単位の
    # タイムスタンプ（char）を無音で区切ってセグメント化する。
    units = stamps.get("char") or []
    if len(stamps.get("word") or []) > 1:
        units = stamps["word"]

    normalized = [
        {
            "start": u["start"],
            "end": u["end"],
            "word": "".join(u["char"]) if isinstance(u.get("char"), list) else u.get("word", ""),
        }
        for u in units
    ]
    segments = group_words(normalized)
    # 区間ごとに切り出して評価する用途のため、文字単位のタイムスタンプも残す
    # （セグメントは無音でまとめた結果、区間境界をまたぐことがある）
    units = [{"start": u["start"], "end": u["end"], "text": u["word"]} for u in normalized]
    return {
        "text": str(hyp.text),
        "segments": segments,
        "units": units,
        "model": MODEL_ID,
        "elapsed_sec": round(elapsed, 3),
        "audio_sec": round(wav_duration(wav), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--outdir", default="data/outputs")
    args = parser.parse_args()

    import nemo.collections.asr as nemo_asr

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.audio))[0]

    print(f"モデル読み込み: {MODEL_ID}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_ID)
    model.eval()

    wav = to_wav16k(args.audio)
    try:
        for run in range(1, args.runs + 1):
            print(f"--- run {run}/{args.runs} ---")
            result = transcribe(model, wav)
            base = os.path.join(args.outdir, f"{stem}__parakeet__run{run}")

            with open(f"{base}.txt", "w", encoding="utf-8") as fh:
                fh.write(result["text"])
            with open(f"{base}.srt", "w", encoding="utf-8") as fh:
                fh.write(to_srt(result["segments"]))
            with open(f"{base}.json", "w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False, indent=1)

            rtf = result["elapsed_sec"] / result["audio_sec"] if result["audio_sec"] else float("nan")
            print(
                f"  セグメント数: {len(result['segments'])} / 文字数: {len(result['text'])}"
                f" / 転写 {result['elapsed_sec']:.2f}s"
                f"（音声 {result['audio_sec']:.1f}s・実時間比 {rtf:.3f}）"
            )
            print(f"  出力: {base}.txt|.srt|.json")
    finally:
        os.unlink(wav)
    return 0


if __name__ == "__main__":
    sys.exit(main())
