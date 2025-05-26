#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTTM 形式のダイアリゼーション結果を可視化するスクリプト
usage: python plot_rttm_timeline.py /path/to/file.rttm
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_rttm(rttm_path: Path) -> pd.DataFrame:
    """RTTM を DataFrame で読み込む"""
    cols = [
        "type",
        "file",
        "chan",
        "start",
        "dur",
        "NA1",
        "NA2",
        "speaker",
        "NA3",
        "NA4",
    ]
    df = pd.read_csv(
        rttm_path,
        sep=r"\s+",
        names=cols,
        usecols=["start", "dur", "speaker"],
        engine="python",
    )
    return df


def plot_timeline(df: pd.DataFrame, title: str = "Speaker Diarization Timeline"):
    """broken‐bar plot でタイムライン描画"""
    speakers = sorted(df["speaker"].unique())
    colors = plt.cm.get_cmap("tab10", len(speakers))  # 自動で 10 色循環

    fig, ax = plt.subplots(figsize=(12, 1.2 + 0.6 * len(speakers)))

    for idx, spk in enumerate(speakers):
        segs = df[df["speaker"] == spk]
        ax.broken_barh(
            segs[["start", "dur"]].values,
            (idx - 0.4, 0.8),
            facecolors=colors(idx),
        )

    ax.set_xlabel("Time [s]")
    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_rttm_timeline.py /path/to/file.rttm", file=sys.stderr)
        sys.exit(1)

    rttm_path = Path(sys.argv[1]).expanduser()
    # rttm_path = Path("/mnt/kiso-qnap3/yuabe/m1/useWhisperX/data/sample.rttm")
    if not rttm_path.is_file():
        print(f"File not found: {rttm_path}", file=sys.stderr)
        sys.exit(1)

    df = load_rttm(rttm_path)
    plot_timeline(df, title=rttm_path.stem)

    # headless サーバなどでは PNG 保存に切り替える
    out_img = rttm_path.with_suffix(".png")
    plt.savefig(out_img, dpi=200)
    print(f"Saved plot to: {out_img}")


if __name__ == "__main__":
    main()
