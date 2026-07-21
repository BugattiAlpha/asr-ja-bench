"""正解テキストが既知のテスト音声を生成する（CER を客観計算するため）。

Windows SAPI（Microsoft Haruka / ja-JP）でオフライン合成し、ffmpeg で
雑音・音楽を重畳した派生版を作る。実音源では正解が分からず主観判定に頼るしかないが、
合成音声なら投入したテキストがそのまま正解になる。

3パターン（各30秒前後）:
  A clean      : 固有名詞・数字を含む素直な読み上げ（基準）
  B repetition : 実コンテンツとしての反復を含む（反復ループ検出の偽陽性を試す）
  C noisy      : A と同一テキストに音楽・雑音を重畳（反復ループの誘発を試す）

A と C はテキストが同一なので、差分はそのまま「雑音の影響」を意味する。

使い方:
    uv run python scripts/make_test_audio.py [--outdir data/testset]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

# 正解テキスト。読み上げさせる文字列そのものが参照文になる。
#
# ⚠️ 数字は必ずアラビア数字で書くこと。仮名で「いち、に、さん」と書くと、TTS は
# 正しく読み上げ、ASR も正しく「1、2、3」と書き起こすのに、読み正規化では数字が
# 数字のまま残るため全部誤りに数えられ、CER が20文字分ほど不当に悪化する
# （2026-07-20 に実際にこれで測り直した）。「三つ」も同じ理由で「3つ」と書く。
TEXTS: dict[str, str] = {
    "clean": (
        "こんにちは。今日は音声認識の精度をたしかめる実験をします。"
        "京都市と大阪市と名古屋市の3つの町の名前を読み上げます。"
        "数字も読みます。1、2、3、4、5、6、7、8、9、10。"
        "この文章はすべて正解が分かっているので、まちがえた文字の数をかぞえられます。"
        "最後まで正しく書き起こせるかどうかを見ています。"
    ),
    # ⚠️ 反復回数は控えめにすること。8回並べたら TTS が6回しか読まず、正解と音声がずれた
    #    （2026-07-21 実測）。ASR の認識結果を正解に採用するのは循環論法なので不可
    #    ── 反復の数え落としは ASR の既知の弱点そのもの。TTS が確実に読める長さに抑える。
    "repetition": (
        "ここからは、わざと同じ言葉をくりかえします。"
        "ちょん、ちょん、ちょん、ちょん。"
        "これは本当にくりかえしている音声です。まちがいではありません。"
        "ちょちょん、ちょちょん、ちょちょん。"
        "くりかえしが終わったら、ふつうの文章にもどります。これで終わりです。"
    ),
}
TEXTS["noisy"] = TEXTS["clean"]  # C は A と同一テキスト（差分＝雑音の影響）
# D: 別の話者が同時にしゃべる（実音源の失敗条件＝声の重なりを模す）。
# 参照文は主話者のテキスト（＝clean）で、裏の声は妨害として扱う。
TEXTS["overlap"] = TEXTS["clean"]
OVERLAP_INTERFERENCE = (
    "ちがう話をしています。じゃまをするための声です。"
    "うしろでずっとしゃべりつづけています。これは正解にふくみません。"
    "まだしゃべっています。もうすこしつづけます。おわりです。"
)

# 合成音声にかぶせる音楽・雑音。正弦波の和音＋ピンクノイズ。
# amix は既定で入力数に応じて音量を正規化する（normalize=1）ため、雑音を足したつもりが
# 全体が小さくなるだけになる。必ず normalize=0 を指定すること。
NOISE_FILTER = (
    "sine=frequency=440:duration=40[a];"
    "sine=frequency=554:duration=40[b];"
    "sine=frequency=659:duration=40[c];"
    "anoisesrc=duration=40:color=pink:amplitude=0.5[n];"
    "[a][b][c][n]amix=inputs=4:duration=shortest:normalize=0,volume=0.5[bg]"
)


IRODORI_DIR = r"C:\Users\Cooliris\MY_Work_Space\IrodoriTTS"
IRODORI_REF = os.path.join(IRODORI_DIR, "outputs", "asr_bench", "ref_voice.wav")
IRODORI_BASE_REPO = "Aratako/Irodori-TTS-500M-v2"


def _split_sentences(text: str) -> list[str]:
    """句点で文に割る。30秒を一度に合成すると崩れうるため文単位で回す。"""
    parts = re.split(r"(?<=。)", text)
    return [p.strip() for p in parts if p.strip()]


def synthesize_irodori(text: str, dst: str) -> None:
    """Irodori-TTS（MIT）で合成する。VoiceDesign で作った基準声をクローン元にする。

    実在人物の声を複製しないよう、参照音声はキャプションから生成した合成声のみを使う。
    ⚠️ base モデルの ref 付き推論は fp32 必須（bf16 は ref latent と dtype 不一致で落ちる）。
    """
    if not os.path.exists(IRODORI_REF):
        raise RuntimeError(f"基準声が無い: {IRODORI_REF}（先に VoiceDesign で生成する）")

    with tempfile.TemporaryDirectory() as tmp:
        chunks = []
        for i, sentence in enumerate(_split_sentences(text)):
            out = os.path.join(tmp, f"{i:03d}.wav")
            cmd = [
                "uv", "run", "python", "infer.py",
                "--hf-checkpoint", IRODORI_BASE_REPO,
                "--ref-wav", IRODORI_REF,
                "--text", sentence,
                "--output-wav", out,
                "--model-precision", "fp32",
                "--seed", "42",
            ]
            proc = subprocess.run(cmd, cwd=IRODORI_DIR, capture_output=True, text=True)
            if proc.returncode != 0 or not os.path.exists(out):
                raise RuntimeError(f"Irodori 合成失敗 (rc={proc.returncode}): {proc.stderr[-800:]}")
            chunks.append(out)
            print(f"    文 {i + 1}: {sentence[:24]}…")

        # 文の間に 0.35 秒の無音を挟んで連結する
        listing = os.path.join(tmp, "list.txt")
        silence = os.path.join(tmp, "sil.wav")
        run_ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "0.35", silence])
        with open(listing, "w", encoding="utf-8") as fh:
            for i, c in enumerate(chunks):
                fh.write(f"file '{c}'\n")
                if i < len(chunks) - 1:
                    fh.write(f"file '{silence}'\n")
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", listing, "-ac", "1", "-ar", "16000", dst])


def synthesize(text: str, dst: str) -> None:
    """SAPI で ja-JP 音声を wav 出力する。"""
    ps = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('Microsoft Haruka Desktop')
$s.Rate = 0
$s.SetOutputToWaveFile('{dst}')
$s.Speak(@'
{text}
'@)
$s.Dispose()
"""
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError(f"SAPI 合成に失敗: rc={proc.returncode} {proc.stderr[-500:]}")


def to_16k_mono(src: str, dst: str) -> None:
    run_ffmpeg(["-i", src, "-ac", "1", "-ar", "16000", dst])


def add_noise(src: str, dst: str) -> None:
    """音楽（正弦波の和音）＋ピンクノイズを重畳する（normalize=0 で音量を保つ）。"""
    run_ffmpeg(
        [
            "-i", src,
            "-filter_complex",
            f"{NOISE_FILTER};[0:a][bg]amix=inputs=2:duration=first:normalize=0",
            "-ac", "1", "-ar", "16000", dst,
        ]
    )


def add_overlap(src: str, interference: str, dst: str) -> None:
    """裏で別の発話を同時に流す（声の重なりを再現）。"""
    run_ffmpeg(
        [
            "-i", src, "-i", interference,
            "-filter_complex",
            "[1:a]volume=0.8[bg];[0:a][bg]amix=inputs=2:duration=first:normalize=0",
            "-ac", "1", "-ar", "16000", dst,
        ]
    )


def run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(["ffmpeg", "-y", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 失敗 (rc={proc.returncode}): {proc.stderr[-800:]}")


def duration_of(path: str) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    return float(proc.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data/testset")
    parser.add_argument(
        "--engine", choices=["irodori", "sapi"], default="irodori",
        help="irodori: Irodori-TTS(MIT・記事掲載可) / sapi: Windows内蔵(掲載不可・測定専用)",
    )
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    speak = synthesize_irodori if args.engine == "irodori" else synthesize
    print(f"エンジン: {args.engine}")

    manifest = {}
    for name, text in TEXTS.items():
        final = os.path.join(args.outdir, f"{name}.wav")
        if os.path.exists(final):
            print(f"[{name}] 既存を再利用")
            manifest[name] = {
                "audio": final, "reference_chars": len(text),
                "seconds": round(duration_of(final), 2),
            }
            continue

        # noisy / overlap は clean と同一テキスト。合成は決定的（seed固定）なので
        # 生成済みの clean をそのまま土台に使う。作り直すと1本15分かかるうえ、
        # 流用した方が「同一音声に雑音/重なりだけを足した」ことを厳密に保証できる。
        base_of_clean = name in ("noisy", "overlap")
        clean_wav = os.path.join(args.outdir, "clean.wav")
        raw = os.path.join(args.outdir, f"{name}_raw.wav")
        print(f"[{name}] {'clean を土台に加工' if base_of_clean else '合成中'}…")

        if base_of_clean and os.path.exists(clean_wav):
            raw = clean_wav
        else:
            speak(text, os.path.abspath(raw))

        if name == "noisy":
            add_noise(raw, final)
        elif name == "overlap":
            interference = os.path.join(args.outdir, "_interference.wav")
            if not os.path.exists(interference):
                speak(OVERLAP_INTERFERENCE, os.path.abspath(interference))
            add_overlap(raw, interference, final)
            os.remove(interference)
        else:
            to_16k_mono(raw, final)
        if raw != clean_wav and os.path.exists(raw):
            os.remove(raw)

        with open(os.path.join(args.outdir, f"{name}.reference.txt"), "w", encoding="utf-8") as fh:
            fh.write(text)

        seconds = duration_of(final)
        manifest[name] = {"audio": final, "reference_chars": len(text), "seconds": round(seconds, 2)}
        print(f"{name}: {seconds:.1f}秒 / 正解 {len(text)}文字 -> {final}")

    with open(os.path.join(args.outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
