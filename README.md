# asr-ja-bench

日本語の文字起こしモデルを、**正解テキストが分かる音声**で比較するための検証コード一式。

`nvidia/parakeet-tdt_ctc-0.6b-ja` と `faster-whisper large-v3` を、同じハーネス・同じ CER 計算で
測るために書いた。台本と音声を差し替えれば、自分の音源でそのまま測れる。

記事: 「日本語の文字起こし、Whisperより良いモデルはあるのか」（bugattialpha.com）

## できること

- 音声合成で**正解付きのテスト音声**を作る（素直な読み上げ / 意図的な反復 / 音楽とノイズ / 声の重なり）
- 日本語向けに正規化した **CER**（文字誤り率）を計算する
- 両モデルを同じ条件で複数回実行し、結果を集計する
- 反復ループ（同じ語で埋まる現象）を検出する

## 必要なもの

- Windows / CUDA GPU（RTX 2070 SUPER・VRAM 8GB で動作を確認）
- Python 3.12、[uv](https://github.com/astral-sh/uv)
- ffmpeg / ffprobe（PATH に通っていること）
- 音声合成を使う場合は [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)（MIT）

```bash
uv sync
```

## 使い方

```bash
# 1. テスト音声を作る（正解テキストも一緒に出力される）
uv run python scripts/make_test_audio.py --engine irodori

# 2. 両モデルで文字起こしする（3回ずつ）
uv run python scripts/run_parakeet.py data/testset/clean.wav --runs 3 --outdir data/results
uv run python scripts/run_whisper.py  data/testset/clean.wav --runs 3 --outdir data/results
uv run python scripts/run_whisper.py  data/testset/clean.wav --runs 3 --vad --outdir data/results

# 3. 集計する
uv run python scripts/compare.py
```

読み上げ原稿がある実録音を評価する場合は `scripts/compare_mictest.py` を使う。

## スクリプト

| ファイル | 役割 |
|---|---|
| `scripts/make_test_audio.py` | 正解付きテスト音声の生成（`--engine irodori\|sapi`）|
| `scripts/run_parakeet.py` | NeMo で文字起こし。txt / srt / json を出力 |
| `scripts/run_whisper.py` | faster-whisper で文字起こし。`--vad` で VAD フィルタ |
| `scripts/cer.py` | 日本語向けに正規化した CER |
| `scripts/metrics.py` | 反復スコア・ウィンドウループ検出・かな正規化 |
| `scripts/compare.py` | 合成音声セットの集計 |
| `scripts/compare_mictest.py` | 読み上げ原稿つき実録音の評価 |

## CER の正規化について

日本語では表記のゆれがそのまま「誤り」に化けるので、比較の前に次を畳んでいる。

- **かな漢字**: pykakasi で読みに変換（「京都市」と「きょうとし」を同一視）
- **数字表記**: 台本側をアラビア数字で書く前提（仮名で書くと不当に悪化する）
- **大文字小文字**: `AI` と `ai` を同一視
- **句読点・空白**: 除去

**長音符「ー」は残す。** 句読点と違って音韻的に弁別的で、落とすと「ビール」と「ビル」が同じになり、
ASR がよくやる長音の脱落が計上されなくなる。

代償として**同音異義語の取り違えを見逃す**（「橋」「箸」「端」がすべて「はし」になる）。
この正規化は独自なので、**モデルカードが公表している CER と直接は比較できない**。

正規化のテストは `tests/test_cer.py` にある。

```bash
uv run pytest
```

## 実行環境とデコード設定

- Windows 11 / RTX 2070 SUPER (VRAM 8GB)
- faster-whisper 1.2.1 / `large-v3` / `compute_type=float16` / `beam_size=5` / `language="ja"`
  / `condition_on_previous_text=False` / `no_speech_threshold=0.6` / `log_prob_threshold=-1.0`
  / `compression_ratio_threshold=2.4`
- NeMo 2.7.3 / `nvidia/parakeet-tdt_ctc-0.6b-ja`
- torch 2.13.0+cu126 / torchaudio 2.11.0+cu126（2026-07-28 に cu124 / torch 2.6.0 から更新）

CUDA チャンネルを上げても **CER は変わらなかった**（合成音声4条件は完全一致、実録音は
`v2_broadcast`×turbo の1組だけが動いたが、この組は旧環境でも実行ごとに 0.2996〜0.3230 と
揺れるもので、新環境も同じ帯に収まる）。速度は parakeet だけ約16%速くなり
（同日・同条件で 1.01秒 → 0.87秒）、CTranslate2 で動く whisper 系3つは変わらない。

モデルは `HF_HOME` のキャッシュへ落ちる。

## NeMo を Windows に入れるとき詰まるところ

素直に `uv add "nemo_toolkit[asr]"` を叩くと、4回別々の理由で失敗した。

**1. texterrors がビルドできない**

NeMo が固定する `texterrors==0.5.1` はソース配布しかなく、C 拡張のビルドに Visual Studio が要る。
WER 採点用のライブラリだから外せると思ったら、`nemo.collections.asr` のインポート連鎖に入っていて
外せなかった。wheel が存在する 1.1.8 で上書きして通した。

```toml
[tool.uv]
override-dependencies = ["texterrors==1.1.8"]
```

**2. PyTorch のインデックスを既定にすると、無関係な解決が壊れる**

CUDA 版 torch のためにインデックスを足すと、以後 `setuptools` のような共通パッケージまで
そこから探しにいって解決に失敗する。torch 系だけに限定する。

```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
torchaudio = { index = "pytorch-cu126" }
```

**3. onnx と ml_dtypes の不整合** — `AttributeError: module 'ml_dtypes' has no attribute 'float4_e2m1fn'`。
`uv add "ml_dtypes>=0.5.1"` で解決。

**4. datasets と pyarrow の不整合** — `AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'`。
`uv add "datasets>=3.0"` で解決。

## 日本語固有の落とし穴

分かち書きしない言語なので、NeMo で**単語単位のタイムスタンプを取ると全文が1語**として返ってくる。
句読点もほとんど出力されないため、句読点ベースのセグメント分割も機能しない。
文字単位のタイムスタンプ（`timestamp['char']`）を無音で区切ってセグメントを作っている。

## 音声とデータについて

生成した音声・文字起こし結果・録音ファイルはリポジトリに含めていない（`.gitignore`）。
`make_test_audio.py` を実行すれば再生成できる。

テスト音声の生成に使った Irodori-TTS は MIT ライセンス。生成音声の公開に制限はないが、
モデルカードには「本人の同意なく個人の声をクローンしない」「誤解を招くディープフェイクに使わない」
という倫理的制限がある。

## ライセンス

MIT License. 詳細は [LICENSE](LICENSE) を参照。
