# transcribe-whisper-qwen

日本語音声ファイルを文字起こしするツールです。Whisper を使用して音声をテキスト化し、Qwen などの大規模言語モデル（LLM）との連携も想定しています。

## 機能

- Whisper による高精度な日本語音声認識
- ローカルファイル対応（例: `.mp3`, `.wav`, `.m4a`）
- Qwenなど他モデルとの接続用拡張性を考慮した構成

## ファイル構成

```
transcribe-whisper-qwen/
├── README.md           # このファイル
├── LICENSE             # MITライセンス
├── requirements.txt    # 依存パッケージ一覧
├── .gitignore          # Gitで無視するファイル
└── transcribe.py       # メインスクリプト
```

## セットアップ方法

```bash
git clone https://github.com/z39084yu9023jr/transcribe-whisper-qwen.git
cd transcribe-whisper-qwen
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 使用方法

```bash
python transcribe.py --input audio.m4a
```

## ライセンス

MIT License
