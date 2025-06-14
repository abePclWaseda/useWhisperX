import os
import json
from pathlib import Path
from dotenv import load_dotenv
import whisperx
import torch

load_dotenv()

# WhisperX設定
device = "cuda"
compute_type = "float16"
auth_token = os.getenv("HUGGINGFACE_AUTH_TOKEN")

# 対象ファイル
file_id = "52a89d2d9aab22588117a6b8599add47"
wav_path = Path(
    f"/mnt/work-qnap/llmc/J-CHAT/audio/podcast_test/00000-of-00001/cuts.000000/{file_id}.wav"
)
json_path = Path(
    f"/mnt/kiso-qnap3/yuabe/m1/useReazonSpeech/data/text_nemo/{file_id}.json"
)
output_path = Path(f"data/text_nemo_whisperx/{file_id}.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Whisper モデルとアライメントモデルをロード
asr_model = whisperx.load_model("large-v2", device, compute_type=compute_type)
align_model, metadata = whisperx.load_align_model("ja", device)
diarize_model = whisperx.diarize.DiarizationPipeline(
    use_auth_token=auth_token, device=device
)

# 話者名マッピング
speaker_map = {
    "SPEAKER_00": "A",
    "SPEAKER_01": "B",
}

try:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    # JSON読み込み
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])

    # 音声読み込み
    audio = whisperx.load_audio(str(wav_path))

    # アライメント
    aligned_result = whisperx.align(
        segments, align_model, metadata, audio, device, return_char_alignments=False
    )

    # 話者分離
    diarize_segments = diarize_model(str(wav_path), num_speakers=2)

    # チェックポイント：NaNや空データがないか確認
    print(f"[DEBUG] diarize_segments for {wav_path.name}:")
    print(diarize_segments.head())
    print(diarize_segments.dtypes)
    print(diarize_segments.isnull().sum())

    # 話者割当て
    if aligned_result.get("segments") and not diarize_segments.empty:
        result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
    else:
        raise ValueError("Empty segments or diarization output")

    # 整形
    formatted = []
    for segment in result["segments"]:
        backup_speaker = segment.get("speaker", "Unknown")
        for w in segment.get("words", []):
            speaker = speaker_map.get(w.get("speaker", backup_speaker), "Unknown")
            formatted.append(
                {
                    "speaker": speaker,
                    "word": w["word"],
                    "start": w["start"],
                    "end": w["end"],
                }
            )

    # 保存
    with output_path.open("w", encoding="utf-8") as jf:
        json.dump(formatted, jf, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] Processed {file_id}, saved to {output_path}")

except Exception as e:
    print(f"[ERROR] Failed to process {file_id}: {e}")
