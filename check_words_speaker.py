import json, pandas as pd, pathlib, textwrap

# --- ① 入力ファイル（さきほど保存した JSON） ---
json_path = pathlib.Path(
    "/mnt/kiso-qnap3/yuabe/m1/useWhisperX/data/0b10fe56c17e068fcca9ef0d470e6800_large.json"
)

# --- ② 読み込み & フラット化 ---
rows = []
with json_path.open(encoding="utf-8") as f:
    for seg in json.load(f):
        for w in seg["words"]:
            rows.append(
                {
                    "start": w["start"],
                    "end": w["end"],
                    "speaker": w.get("speaker", "UNK"),
                    "word": w["word"],
                    "seg_id": f"{seg['start']:.3f}-{seg['end']:.3f}",
                    "conf": w.get("score", None),
                }
            )
df = pd.DataFrame(rows)

# --- ③ CSV で保存（Excel などで確認する用） ---
csv_path = json_path.with_suffix(".words.csv")
df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"CSV を保存しました → {csv_path}")

# --- ④ ざっと内容をテキストで確認（上位 30 行） ---
head_txt = df.head(30).to_string(
    index=False,
    formatters={
        "start": "{:.3f}".format,
        "end": "{:.3f}".format,
    },
)
txt_path = json_path.with_suffix(".head.txt")
txt_path.write_text(head_txt, encoding="utf-8")
print(f"先頭 30 行を {txt_path} に出力しました\n")
print(textwrap.indent(head_txt, "  "))
