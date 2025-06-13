import os
import whisperx
import torch
import json
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# WhisperX設定
device = "cuda"
compute_type = "float16"
batch_size = 16
auth_token = os.getenv("HUGGINGFACE_AUTH_TOKEN")

# 入出力ディレクトリ
file = "/mnt/kiso-qnap3/yuabe/m1/useAsteroid/data/J-CHAT/audio/podcast_test/00000-of-00001/cuts.000000/dcdd979f47cb788aeb8ef58033d37fff.wav"
output_dir = "data/json"
os.makedirs(output_dir, exist_ok=True)

# Whisperモデルロード
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

audio_path = os.path.join(file)
audio = whisperx.load_audio(audio_path)

# 1. 音声認識
result = model.transcribe(audio, batch_size=batch_size, language="ja")

with open("data/text/result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# 2. アライメント
model_a, metadata = whisperx.load_align_model(result["language"], device)
result = whisperx.align(
    result["segments"],
    model_a,
    metadata,
    audio,
    device,
    return_char_alignments=False,
)

with open("data/text/result_align.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# 3. 話者分離
diarize_model = whisperx.diarize.DiarizationPipeline(
    use_auth_token=auth_token, device=device
)
diarize_segments = diarize_model(audio_path)
result = whisperx.assign_word_speakers(diarize_segments, result)

with open("data/text/result_diarize.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# # 4. 整形＆保存
# speaker_map = {
#     "SPEAKER_00": "A",
#     "SPEAKER_01": "B",
# }

# formatted = []
# for segment in result["segments"]:
#     backup_speaker = segment.get("speaker", "Unknown")
#     for w in segment.get("words", []):
#         speaker = speaker_map.get(w.get("speaker", backup_speaker), "Unknown")
#         formatted.append(
#             {
#                 "speaker": speaker,
#                 "word": w["word"],
#                 "start": w["start"],
#                 "end": w["end"],
#             }
#         )

# # 出力ファイル名
# json_path = Path(output_dir) / f"{Path(file).stem}.json"
# with json_path.open("w", encoding="utf-8") as jf:
#     jf.write("[\n")
#     for i, entry in enumerate(formatted):
#         jf.write("    ")
#         json.dump(entry, jf, ensure_ascii=False)
#         jf.write(",\n" if i != len(formatted) - 1 else "\n")
#     jf.write("]\n")
