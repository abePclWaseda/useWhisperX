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
output_dir = Path(
    "/mnt/kiso-qnap3/yuabe/m1/moshi-finetune/data/J-CHAT/text/podcast_test/00000-of-00001/cuts.000000"
)
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

# 話者分離失敗ファイルを記録
failed_files = []

# ファイルごとに処理
for wav_path in tqdm(sorted(wav_dir.glob("*.wav"))):
    base_name = wav_path.stem
    json_path = json_nemo_dir / f"{base_name}.json"
    output_path = output_dir / f"{base_name}.json"

    if not json_path.exists():
        print(f"[!] JSON not found for {base_name}, skipping.")
        failed_files.append(wav_path.name)
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

        # (1) 話者が 2 つ検出されたか？
        if diarize_segments["speaker"].nunique() != 2:
            print(f"[SKIP] Detected speakers ≠ 2 for {wav_path.name}")
            failed_files.append(wav_path.name)
            continue

        # (2) 話者割当て
        result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
        if not result or not result.get("segments"):
            print(f"[SKIP] Speaker assignment empty for {wav_path.name}")
            failed_files.append(wav_path.name)
            continue

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

        speakers_in_file = {item["speaker"] for item in formatted}
        if speakers_in_file != {"A", "B"}:
            print(f"[SKIP] Only {speakers_in_file} detected in {wav_path.name}")
            failed_files.append(wav_path.name)
            continue
        # 保存
        with output_path.open("w", encoding="utf-8") as jf:
            # 1 行 JSON 文字列をリスト化
            indent = "  "  # ← ここを変えればタブや 2 文字分も可
            json_lines = [
                indent + json.dumps(obj, ensure_ascii=False) for obj in formatted
            ]

            # [] 付きで書き込む
            jf.write("[\n")
            jf.write(",\n".join(json_lines))
            jf.write("\n]")

    except Exception as e:
        print(f"[ERROR] Failed to process {base_name}: {e}")
        failed_files.append(wav_path.name)
        continue

# ログファイル出力
log_path = Path("failed_speaker_assignment.txt")
with log_path.open("w", encoding="utf-8") as log_file:
    for fname in failed_files:
        log_file.write(f"{fname}\n")

print(
    f"[INFO] {len(failed_files)} files failed speaker assignment. Log saved to {log_path}"
)
