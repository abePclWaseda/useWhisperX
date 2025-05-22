# whisperxモジュールから必要な関数やクラスをインポート
import whisperx

# 時間の計算に使用するためのtimedeltaクラスをインポート
from datetime import timedelta

# 進捗バーの表示に使用するtqdmモジュールをインポート
from tqdm import tqdm

# 環境変数の読み込み用
from dotenv import load_dotenv
import os

# .env ファイルを読み込む
load_dotenv()

# 使用するデバイス（GPU）を指定
device = "cuda"
# 入力となる音声ファイルのパスを指定
audio_file = "devon.mp3"
# バッチサイズを指定（GPUメモリが不足している場合は数を減らす）
batch_size = 16
# 計算の精度を指定（GPUメモリが不足している場合は"int8"に変更可能、ただし精度は低下する可能性あり）
compute_type = "float16"
# Hugging Faceの認証トークンを指定
# 環境変数からトークンを取得
auth_token = os.getenv("HUGGINGFACE_AUTH_TOKEN")

# Whisper ASRモデルを指定のデバイスと精度で読み込む
model = whisperx.load_model("small", device, compute_type=compute_type)
# 入力音声を読み込む
audio = whisperx.load_audio(audio_file)
# Whisper ASRモデルを使用して音声をテキストに変換し、結果を取得
result = model.transcribe(audio, batch_size=batch_size)

# 言語に応じたアラインメントモデルを読み込む
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device
)
# Whisperのアラインメントモデルを使用して音声とテキストのアラインメントを行う
result = whisperx.align(
    result["segments"], model_a, metadata, audio, device, return_char_alignments=False
)

# DiarizationPipelineを使用して音声から話者の情報を取得
diarize_model = whisperx.DiarizationPipeline(use_auth_token=auth_token, device=device)
diarize_segments = diarize_model(audio_file)
# 取得した話者の情報を元に、テキストセグメントに話者情報を割り当てる
result = whisperx.assign_word_speakers(diarize_segments, result)

# 結果をテキストファイルに保存する
with open("result.txt", "w", encoding="utf-8") as f:
    # 各セグメントの情報を処理しながらファイルに書き込む
    for segment in tqdm(result["segments"], desc="Processing segments", ncols=75):
        # セグメントの開始時間を取得して文字列に変換
        start_time = str(timedelta(seconds=segment["start"]))
        # セグメントの終了時間を取得して文字列に変換
        end_time = str(timedelta(seconds=segment["end"]))
        # セグメントの話者情報を取得
        speaker = segment["speaker"]
        # セグメントのテキストを取得
        text = segment["text"]
        # セグメントの情報をファイルに書き込む
        f.write(f"{start_time}-{end_time}\n{speaker}\n{text}\n\n")
