import librosa
import numpy as np
import itertools

# 波形とスピーカーのラベルをプロット
import matplotlib.pyplot as plt

# 波形プロット
plt.figure(figsize=(15, 5))
plt.plot(np.arange(len(waveform)) / 22050, waveform, label="Waveform")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Waveform and Speaker Diarization")

# スピーカーごとに色を割り当てる
speaker_colors = {}
for _, _, speaker in diarization.itertracks(yield_label=True):
    if speaker not in speaker_colors:
        speaker_colors[speaker] = plt.cm.tab10(len(speaker_colors) % 10)

# スピーカーラベルのプロット（色分け）
for segment, _, speaker in diarization.itertracks(yield_label=True):
    plt.axvspan(
        segment.start,
        segment.end,
        alpha=0.3,
        color=speaker_colors[speaker],
        label=speaker,
    )

# 凡例の重複を削除
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()
import IPython.display as ipd

ipd.display(ipd.Audio(waveform, rate=22050))
