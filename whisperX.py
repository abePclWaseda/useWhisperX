import whisperx
from datetime import timedelta
from pathlib import Path
from tqdm import tqdm
import whisperx.diarize
from dotenv import load_dotenv
import os
import json

load_dotenv()

device = "cuda"
audio_file = "/mnt/kiso-qnap3/yuabe/m1/useWhisperX/data/audio/0b10fe56c17e068fcca9ef0d470e6800.wav"
batch_size = 16
compute_type = "float16"
auth_token = os.getenv("HUGGINGFACE_AUTH_TOKEN")

model = whisperx.load_model("large-v2", device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)

model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device
)

result = whisperx.align(
    result["segments"], model_a, metadata, audio, device, return_char_alignments=False
)

diarize_model = whisperx.diarize.DiarizationPipeline(
    use_auth_token=auth_token, device=device
)
diarize_segments = diarize_model(audio_file)
result = whisperx.assign_word_speakers(diarize_segments, result)

output_path = Path(
    "/mnt/kiso-qnap3/yuabe/m1/useWhisperX/data/0b10fe56c17e068fcca9ef0d470e6800_large.txt"
)
output_path.parent.mkdir(parents=True, exist_ok=True)

json_path = output_path.with_suffix(".json")

with json_path.open("w", encoding="utf-8") as jf:
    json.dump(result["segments"], jf, ensure_ascii=False, indent=2)

# with output_path.open(
#     "w",
#     encoding="utf-8",
# ) as f:
#     for segment in tqdm(result["segments"], desc="Processing segments", ncols=75):
#         start_time = str(timedelta(seconds=segment["start"]))
#         end_time = str(timedelta(seconds=segment["end"]))
#         speaker = segment["speaker"]
#         text = segment["text"]
#         f.write(f"{start_time}-{end_time}\n{speaker}\n{text}\n\n")
# for w in segment["words"]:  # まずは，30秒単位で．
#     start_time = str(timedelta(seconds=w["start"]))
#     end_time = str(timedelta(seconds=w["end"]))
#     speaker = w.get("speaker", "UNK")
#     text = w["word"]
#     f.write(f"{start_time}-{end_time}\n{speaker}\n{text}\n\n")
