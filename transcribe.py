#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Qwen3:14b を使用した日本語音声文字起こしツール (Windows版)
対応形式: MP3, MP4, MKV
"""

import os
import sys
import subprocess
import tempfile
import requests
import json
import argparse
from pathlib import Path
import time
import platform
import shutil

class JapaneseSpeechTranscriber:
    def __init__(self, ollama_url="http://localhost:11434", model="qwen3:14b", chunk_duration=600):
        """
        初期化
        
        Args:
            ollama_url (str): Ollama サーバーのURL
            model (str): 使用するモデル名
            chunk_duration (int): チャンク分割時間（秒）、0なら分割しない
        """
        self.ollama_url = ollama_url
        self.model = model
        self.chunk_duration = chunk_duration  # デフォルト10分
        self.supported_formats = ['.mp3', '.mp4', '.mkv']
        
    def check_dependencies(self):
        """必要なツールがインストールされているかチェック"""
        required_tools = []
        missing_tools = []
        
        # ffmpeg のチェック
        ffmpeg_found = False
        for ffmpeg_name in ['ffmpeg', 'ffmpeg.exe']:
            if shutil.which(ffmpeg_name):
                ffmpeg_found = True
                break
        if not ffmpeg_found:
            missing_tools.append('ffmpeg')
            
        # ffprobe のチェック  
        ffprobe_found = False
        for ffprobe_name in ['ffprobe', 'ffprobe.exe']:
            if shutil.which(ffprobe_name):
                ffprobe_found = True
                break
        if not ffprobe_found:
            missing_tools.append('ffprobe')
            
        # whisper のチェック
        whisper_found = False
        for whisper_name in ['whisper', 'whisper.exe']:
            if shutil.which(whisper_name):
                whisper_found = True
                break
        if not whisper_found:
            missing_tools.append('whisper')
        
        if missing_tools:
            print("以下のツールがインストールされていません:")
            for tool in missing_tools:
                print(f"  - {tool}")
            print("\nWindowsでのインストール方法:")
            print("# ffmpeg のインストール")
            print("  1. https://ffmpeg.org/download.html からダウンロード")
            print("  2. または Chocolatey: choco install ffmpeg")
            print("  3. または winget: winget install FFmpeg")
            print("  4. または scoop: scoop install ffmpeg")
            print("# Whisper のインストール")
            print("  pip install openai-whisper")
            print("# Ollama のインストール")
            print("  https://ollama.ai/ からダウンロードしてインストール")
            print("\n注意: インストール後は新しいコマンドプロンプトを開いてください")
            return False
        return True
    
    def check_ollama_connection(self):
        """Ollama サーバーへの接続をチェック"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                print(f"利用可能なモデル: {', '.join(model_names) if model_names else 'なし'}")
                
                if self.model not in model_names:
                    print(f"警告: モデル '{self.model}' が見つかりません")
                    if model_names:
                        print(f"代わりに利用可能なモデル: {', '.join(model_names)}")
                        # 利用可能なモデルがあれば最初のものを使用
                        self.model = model_names[0]
                        print(f"代わりに '{self.model}' を使用します")
                    else:
                        print("利用可能なモデルがありません。先にモデルをダウンロードしてください:")
                        print("  ollama pull qwen3:14b")
                        return False
                return True
            else:
                print(f"Ollama サーバーに接続できません: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("Ollama サーバーが起動していません。")
            print("コマンドプロンプトで 'ollama serve' を実行してサーバーを起動してください。")
            return False
        except requests.exceptions.Timeout:
            print("Ollama サーバーへの接続がタイムアウトしました。")
            return False
    
    def get_media_duration(self, input_file):
        """メディアファイルの長さを取得"""
        # ffprobe の実行可能ファイル名を決定
        ffprobe_cmd = 'ffprobe'
        if not shutil.which('ffprobe') and shutil.which('ffprobe.exe'):
            ffprobe_cmd = 'ffprobe.exe'
            
        cmd = [
            ffprobe_cmd, '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(input_file)
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                creationflags=subprocess.CREATE_NO_WINDOW,
                encoding='utf-8',
                errors='replace'
            )
            if result.returncode == 0:
                try:
                    return float(result.stdout.strip())
                except ValueError:
                    return None
        except Exception as e:
            print(f"動画の長さ取得でエラー: {e}")
        return None
    
    def extract_audio(self, input_file, temp_dir, start_time=None, duration=None):
        """動画ファイルから音声を抽出"""
        if start_time is not None:
            temp_audio = os.path.join(temp_dir, f"extracted_audio_{int(start_time)}.wav")
        else:
            temp_audio = os.path.join(temp_dir, "extracted_audio.wav")
        
        # ffmpeg の実行可能ファイル名を決定
        ffmpeg_cmd = 'ffmpeg'
        if not shutil.which('ffmpeg') and shutil.which('ffmpeg.exe'):
            ffmpeg_cmd = 'ffmpeg.exe'
        
        cmd = [
            ffmpeg_cmd, '-i', str(input_file),
            '-vn',  # 動画ストリームを無視
            '-acodec', 'pcm_s16le',  # WAV形式で出力
            '-ar', '16000',  # サンプリングレート 16kHz
            '-ac', '1',  # モノラル
            '-y',  # 上書き許可
        ]
        
        # 開始時間と長さを指定
        if start_time is not None:
            cmd.extend(['-ss', str(start_time)])
        if duration is not None:
            cmd.extend(['-t', str(duration)])
            
        cmd.append(temp_audio)
        
        if start_time is not None:
            print(f"音声を抽出中... ({int(start_time//60)}:{int(start_time%60):02d} から {duration}秒)")
        else:
            print("音声を抽出中...")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                creationflags=subprocess.CREATE_NO_WINDOW,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode != 0:
                raise Exception(f"音声抽出に失敗: {result.stderr}")
            
        except Exception as e:
            raise Exception(f"音声抽出でエラーが発生: {str(e)}")
        
        return temp_audio
    
    def transcribe_with_whisper(self, audio_file):
        """Whisper を使用して音声をテキストに変換"""
        print("Whisper で音声認識中...")
        
        # whisper の実行可能ファイル名を決定
        whisper_cmd = 'whisper'
        if not shutil.which('whisper') and shutil.which('whisper.exe'):
            whisper_cmd = 'whisper.exe'
        
        # 出力ディレクトリを指定
        output_dir = os.path.dirname(audio_file)
        
        cmd = [
            whisper_cmd, str(audio_file),
            # '--language', 'japanese',
            '--language', 'ja',
            '--model', 'base',
            '--output_format', 'txt',
            '--output_dir', output_dir
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                creationflags=subprocess.CREATE_NO_WINDOW,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode != 0:
                # エラーメッセージを詳細に表示
                print(f"Whisper のstdout: {result.stdout}")
                print(f"Whisper のstderr: {result.stderr}")
                raise Exception(f"Whisper による音声認識に失敗: {result.stderr}")
            
            # Whisper の出力ファイルを読み込み
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            txt_file = os.path.join(output_dir, f"{base_name}.txt")
            
            if os.path.exists(txt_file):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            else:
                raise Exception("Whisper の出力ファイルが見つかりません")
                
        except Exception as e:
            raise Exception(f"Whisper での処理中にエラー: {str(e)}")
    
    def improve_with_ollama(self, raw_text):
        """Ollama Qwen3:14b でテキストを改善"""
        print("Ollama で文章を改善中...")
        
        prompt = f"""以下は音声認識で得られた日本語のテキストです。このテキストを読みやすく、自然な日本語に修正してください。

音声認識されたテキスト:
{raw_text}

修正時の注意点:
1. 誤認識された単語を適切な日本語に修正
2. 句読点を適切に配置
3. 段落分けを行い読みやすくする
4. 不自然な表現を自然な日本語に修正
5. 元の意味を変えずに文章を整える

修正されたテキスト:"""

        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=data,
                timeout=180  # タイムアウトを延長
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                print(f"Ollama API エラー: {response.status_code}")
                print(f"レスポンス: {response.text}")
                return raw_text
                
        except requests.exceptions.Timeout:
            print("Ollama のレスポンスがタイムアウトしました")
            return raw_text
        except Exception as e:
            print(f"Ollama での処理中にエラー: {e}")
            return raw_text
    
    def transcribe_file(self, input_file, output_file=None):
        """メイン処理: ファイルを文字起こし"""
        input_path = Path(input_file)
        
        # ファイルの存在確認
        if not input_path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {input_file}")
        
        # 対応形式の確認
        if input_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"対応していない形式です。対応形式: {', '.join(self.supported_formats)}")
        
        # 出力ファイル名の設定
        if output_file is None:
            output_file = input_path.with_suffix('.txt')
        
        print(f"処理開始: {input_file}")
        start_time = time.time()
        
        # メディアの長さを取得
        media_duration = None
        if input_path.suffix.lower() in ['.mp4', '.mkv']:
            media_duration = self.get_media_duration(str(input_path))
            if media_duration:
                print(f"動画の長さ: {int(media_duration//60)}分{int(media_duration%60)}秒")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                all_transcriptions = []
                
                # 長い動画の場合は分割処理
                if (media_duration and media_duration > self.chunk_duration and 
                    self.chunk_duration > 0 and input_path.suffix.lower() in ['.mp4', '.mkv']):
                    
                    print(f"長い動画のため {self.chunk_duration//60}分ごとに分割して処理します")
                    
                    chunk_start = 0
                    chunk_num = 1
                    total_chunks = int(media_duration // self.chunk_duration) + (1 if media_duration % self.chunk_duration > 0 else 0)
                    
                    while chunk_start < media_duration:
                        remaining = media_duration - chunk_start
                        chunk_duration = min(self.chunk_duration, remaining)
                        
                        print(f"\n--- チャンク {chunk_num}/{total_chunks} 処理中 ---")
                        
                        # 音声抽出
                        audio_file = self.extract_audio(
                            str(input_path), temp_dir, chunk_start, chunk_duration
                        )
                        
                        # Whisper で音声認識
                        raw_text = self.transcribe_with_whisper(audio_file)
                        
                        if raw_text.strip():
                            # Ollama で文章改善
                            improved_text = self.improve_with_ollama(raw_text)
                            
                            all_transcriptions.append({
                                'chunk': chunk_num,
                                'start_time': chunk_start,
                                'duration': chunk_duration,
                                'raw_text': raw_text,
                                'improved_text': improved_text
                            })
                        
                        chunk_start += self.chunk_duration
                        chunk_num += 1
                    
                    # 全体の結果をまとめる
                    final_text = self.merge_transcriptions(all_transcriptions)
                    
                else:
                    # 通常処理（分割なし）
                    if input_path.suffix.lower() in ['.mp4', '.mkv']:
                        audio_file = self.extract_audio(str(input_path), temp_dir)
                    else:
                        audio_file = str(input_path)
                    
                    raw_text = self.transcribe_with_whisper(audio_file)
                    
                    if not raw_text.strip():
                        raise Exception("音声認識結果が空です")
                    
                    print(f"\n--- 音声認識結果 (生) ---")
                    print(raw_text[:200] + "..." if len(raw_text) > 200 else raw_text)
                    
                    improved_text = self.improve_with_ollama(raw_text)
                    final_text = improved_text
                
                # 結果保存
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("=== 音声文字起こし結果 ===\n\n")
                    f.write("--- 最終テキスト ---\n")
                    f.write(final_text)
                    
                    if all_transcriptions:
                        f.write("\n\n--- チャンク別詳細 ---\n")
                        for trans in all_transcriptions:
                            start_min = int(trans['start_time'] // 60)
                            start_sec = int(trans['start_time'] % 60)
                            f.write(f"\n## チャンク {trans['chunk']} ({start_min}:{start_sec:02d}〜)\n")
                            f.write(trans['improved_text'])
                            f.write("\n")
                    
                    f.write(f"\n\n--- 処理情報 ---\n")
                    f.write(f"入力ファイル: {input_file}\n")
                    f.write(f"処理時間: {time.time() - start_time:.2f}秒\n")
                    f.write(f"使用モデル: {self.model}\n")
                    if media_duration:
                        f.write(f"動画の長さ: {int(media_duration//60)}分{int(media_duration%60)}秒\n")
                    if all_transcriptions:
                        f.write(f"チャンク数: {len(all_transcriptions)}\n")
                
                print(f"\n--- 最終結果 ---")
                print(final_text[:500] + "..." if len(final_text) > 500 else final_text)
                print(f"\n処理完了: {output_file}")
                print(f"処理時間: {time.time() - start_time:.2f}秒")
                
            except Exception as e:
                print(f"エラーが発生しました: {e}")
                raise
    
    def merge_transcriptions(self, transcriptions):
        """複数のチャンクの文字起こし結果をマージ"""
        if not transcriptions:
            return ""
        
        # 改善されたテキストを時系列順に結合
        merged_text = ""
        for i, trans in enumerate(transcriptions):
            if i > 0:
                merged_text += "\n\n"
            merged_text += trans['improved_text']
        
        # 全体をさらに Ollama で最終調整
        final_prompt = f"""以下は音声認識を複数の部分に分けて処理し、それぞれを改善したテキストです。
これらを一つの自然で読みやすい文章として最終調整してください。

分割されたテキスト:
{merged_text}

最終調整時の注意点:
1. 各部分の境界を自然につなげる
2. 重複や矛盾する内容があれば修正
3. 全体として一貫性のある文章にする
4. 段落分けを適切に行う

最終テキスト:"""

        try:
            data = {
                "model": self.model,
                "prompt": final_prompt,
                "stream": False
            }
            
            print("全体の最終調整中...")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=data,
                timeout=240  # タイムアウトを延長
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
                
        except Exception as e:
            print(f"最終調整でエラー: {e}")
        
        # エラーの場合は単純結合を返す
        return merged_text

def main():
    # Windows用のコンソール設定
    if platform.system() == 'Windows':
        try:
            # UTF-8でコンソール出力を設定
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        except:
            pass
    
    parser = argparse.ArgumentParser(
        description="Ollama Qwen3:14b を使用した日本語音声文字起こしツール (Windows版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python transcribe.py input.mp3
  python transcribe.py input.mp4 -o output.txt
  python transcribe.py input.mkv --model qwen3:7b
  
Windows での注意点:
  - ファイルパスにスペースがある場合は引用符で囲んでください
  - 例: python transcribe.py "C:\\Videos\\my video.mp4"
  - ffmpeg, whisper, ollama が正しくインストールされている必要があります
        """
    )
    
    parser.add_argument('input_file', help='入力ファイル (MP3, MP4, MKV)')
    parser.add_argument('-o', '--output', help='出力ファイル (デフォルト: 入力ファイル名.txt)')
    parser.add_argument('--model', default='qwen3:14b', help='Ollama モデル名 (デフォルト: qwen3:14b)')
    parser.add_argument('--ollama-url', default='http://localhost:11434', 
                       help='Ollama サーバーのURL (デフォルト: http://localhost:11434)')
    parser.add_argument('--chunk-duration', type=int, default=600, 
                       help='長い動画の分割時間（秒）、0で分割しない (デフォルト: 600秒=10分)')
    parser.add_argument('--no-chunk', action='store_true', 
                       help='動画を分割せずに一度に処理する')
    
    args = parser.parse_args()
    
    # --no-chunk が指定された場合は chunk_duration を 0 に設定
    chunk_duration = 0 if args.no_chunk else args.chunk_duration
    
    print("=== Windows版 日本語音声文字起こしツール ===")
    print(f"入力ファイル: {args.input_file}")
    print(f"使用モデル: {args.model}")
    print(f"Ollama URL: {args.ollama_url}")
    
    # Transcriber インスタンス作成
    transcriber = JapaneseSpeechTranscriber(
        ollama_url=args.ollama_url,
        model=args.model,
        chunk_duration=chunk_duration
    )
    
    # 依存関係チェック
    print("\n--- 依存関係チェック ---")
    if not transcriber.check_dependencies():
        input("Enterキーを押して終了してください...")
        sys.exit(1)
    
    # Ollama 接続チェック
    print("\n--- Ollama 接続チェック ---")
    if not transcriber.check_ollama_connection():
        input("Enterキーを押して終了してください...")
        sys.exit(1)
    
    try:
        print("\n--- 文字起こし開始 ---")
        # 文字起こし実行
        transcriber.transcribe_file(args.input_file, args.output)
        
        print("\n--- 処理完了 ---")
        input("Enterキーを押して終了してください...")
        
    except KeyboardInterrupt:
        print("\n処理が中断されました")
        input("Enterキーを押して終了してください...")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: {e}")
        input("Enterキーを押して終了してください...")
        sys.exit(1)

if __name__ == "__main__":
    main()
