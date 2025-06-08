from pyannote.audio import Pipeline
from pydub import AudioSegment
import os

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization", use_auth_token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)

wav_path = "input.wav"
diarization = pipeline(wav_path)
audio = AudioSegment.from_file(wav_path)
duration_ms = len(audio)
base = os.path.splitext(wav_path)[0]

# スピーカー2人前提で初期化
speaker_segments = {
    "A": AudioSegment.silent(duration=duration_ms),
    "B": AudioSegment.silent(duration=duration_ms),
}

# 話者IDをA/Bに変換（2人想定）
speaker_map = {}
for turn, _, speaker in diarization.itertracks(yield_label=True):
    if speaker not in speaker_map:
        speaker_map[speaker] = "A" if "A" not in speaker_map.values() else "B"
    mapped_speaker = speaker_map[speaker]

    seg_audio = audio[int(turn.start * 1000) : int(turn.end * 1000)]
    speaker_segments[mapped_speaker] = speaker_segments[mapped_speaker].overlay(
        seg_audio, position=int(turn.start * 1000)
    )

# ステレオ化（L: A, R: B）
stereo_audio = AudioSegment.from_mono_audiosegments(
    speaker_segments["A"], speaker_segments["B"]
)
stereo_audio.export(f"{base}_stereo.wav", format="wav")
print(f"保存: {base}_stereo.wav")
