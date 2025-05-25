import whisperx
from datetime import timedelta
from pathlib import Path
from tqdm import tqdm
import whisperx.diarize
from dotenv import load_dotenv
import os

load_dotenv()

device = "cuda"
audio_file = "/mnt/kiso-qnap3/yuabe/m1/useWhisperX/data/audio/0b10fe56c17e068fcca9ef0d470e6800.wav"
batch_size = 16
compute_type = "float16"
auth_token = os.getenv("HUGGINGFACE_AUTH_TOKEN")

model = whisperx.load_model("tiny", device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)

model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device
)

result = whisperx.align(
    result["segments"], model_a, metadata, audio, device, return_char_alignments=False
)

diarize_model = whisperx.diarize.DiarizationPipeline(
    use_auth_token=auth_token, device="cpu"
)
diarize_segments = diarize_model(audio_file)
result = whisperx.assign_word_speakers(diarize_segments, result)

output_path = Path(
    "/mnt/kiso-qnap3/yuabe/m1/useWhisperX/data/0b10fe56c17e068fcca9ef0d470e6800.txt"
)
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as f:
    for segment in result["segments"]:
        for word_info in segment["words"]:
            w = word_info["word"]
            start = word_info["start"]
            end = word_info["end"]
            spkr = word_info.get("speaker", segment.get("speaker", "UNK"))
            f.write(f"{start:.3f}-{end:.3f}\t{spkr}\t{w}\n")
