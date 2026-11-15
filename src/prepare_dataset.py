
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"D": "joy", "L": "sadness", "N": "neutral", "P": "anger", "Z": "Z"}

meta = pd.read_csv(ROOT / "data/metadata.csv", sep="|", header=None,
                   names=["path", "speaker", "text"], dtype=str,
                   quoting=3, keep_default_na=False)
meta["text"] = meta["text"].str.strip()
meta["label"] = meta["path"].str.split("/").str[1]

stats = {"rows_total": len(meta)}
meta = meta[meta["text"] != ""]
stats["rows_nonempty"] = len(meta)



n_labels = meta.groupby("text")["label"].nunique()
multi = n_labels[n_labels > 1].index
stats["unique_texts_all"] = int(n_labels.size)
stats["multi_label_texts"] = int(len(multi))
stats["multi_label_texts_by_n_labels"] = {int(k): int(v) for k, v in n_labels[n_labels > 1].value_counts().sort_index().items()}
stats["rows_of_multi_label_texts"] = int(meta["text"].isin(multi).sum())

single = meta[~meta["text"].isin(multi)]
stats["samples_before_dedup"] = len(single)
stats["samples_before_dedup_per_class"] = single["label"].value_counts().sort_index().to_dict()

dataset = single.drop_duplicates(subset="text")[["text", "label"]].reset_index(drop=True)
stats["samples_after_dedup"] = len(dataset)
stats["samples_after_dedup_per_class"] = dataset["label"].value_counts().sort_index().to_dict()

dataset.to_csv(ROOT / "data/emo_text_dataset.csv", index=False)
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
print(json.dumps(stats, indent=2, ensure_ascii=False))
