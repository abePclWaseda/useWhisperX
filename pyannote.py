from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv
import json

load_dotenv()
auth_token = os.getenv("HUGGINGFACE_AUTH_TOKEN")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=auth_token
)

# send pipeline to GPU (when available)
import torch

pipeline.to(torch.device("cuda"))

diarization = pipeline(
    "/mnt/kiso-qnap3/yuabe/m1/useWhisperX/data/audio/0b10fe56c17e068fcca9ef0d470e6800.wav"
)

with open("/mnt/kiso-qnap3/yuabe/m1/useWhisperX/data/sample.rttm", "w") as f:
    diarization.write_rttm(f)

print(diarization)
