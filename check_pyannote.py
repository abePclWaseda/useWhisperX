import whisperx, pyannote.audio, torch

print("pyannote version inside WhisperX run :", pyannote.audio.__version__)

# ← ここで WhisperX を普通に実行
model = whisperx.load_model("tiny", device="cuda")
audio = whisperx.load_audio(
    "/mnt/kiso-qnap3/yuabe/m1/useWhisperX/data/audio/0b10fe56c17e068fcca9ef0d470e6800.wav"
)
result = model.transcribe(audio)

# 実行途中に pyannote のモジュールパスを出力してみる
import inspect, os

print("pyannote module path:", inspect.getfile(pyannote.audio))
