import os
import json
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import whisperx
import torch

load_dotenv()

# WhisperX設定
device = "cuda"
compute_type = "float16"
auth_token = os.getenv("HUGGINGFACE_AUTH_TOKEN")

# 入出力ディレクトリ
wav_dir = Path(
    "/mnt/work-qnap/llmc/J-CHAT/audio/podcast_test/00000-of-00001/cuts.000000"
)
json_nemo_dir = Path("/mnt/kiso-qnap3/yuabe/m1/useReazonSpeech/data/text_nemo")
output_dir = Path("data/text_nemo_whisperx")
output_dir.mkdir(parents=True, exist_ok=True)

# Whisper モデルとアライメントモデルを事前ロード
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

# ファイルごとに処理
for wav_path in tqdm(sorted(wav_dir.glob("*.wav"))):
    base_name = wav_path.stem
    json_path = json_nemo_dir / f"{base_name}.json"
    output_path = output_dir / f"{base_name}.json"

    if not json_path.exists():
        print(f"[!] JSON not found for {base_name}, skipping.")
        continue

    try:
        # JSON読み込み
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        segments = data.get("segments", [])

        # 音声ロード
        audio = whisperx.load_audio(str(wav_path))

        # アライメント処理
        aligned_result = whisperx.align(
            segments,
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        # 話者分離
        diarize_segments = diarize_model(str(wav_path), num_speakers=2)
        result = whisperx.assign_word_speakers(diarize_segments, aligned_result)

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

    except Exception as e:
        print(f"[ERROR] Failed to process {base_name}: {e}")
