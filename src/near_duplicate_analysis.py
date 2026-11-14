"""Diagnostic: how much do near-duplicate train/test texts (transcription variants of the same
sentence read by different speakers) inflate the scores? Accuracy by max char-n-gram cosine to train."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from common import DATA, FIG, RES, set_style

set_style()
sp = pd.read_csv(DATA / "train_test_split.csv")
tr = sp[sp.split == "train"].reset_index(drop=True)
pred = pd.read_csv(RES / "test_predictions.csv")  # test texts (unique) in prediction order
te = pred[["text", "emotion"]]
assert set(te.text) == set(sp.loc[sp.split == "test", "text"])

cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2).fit(tr.text)
A, B = cv.transform(te.text), cv.transform(tr.text).T.tocsc()
best, arg = [], []
for s in range(0, A.shape[0], 2000):
    S = (A[s:s + 2000] @ B).toarray()
    arg.append(S.argmax(1)); best.append(S.max(1))
pred["nn_sim"] = np.concatenate(best)
nn = np.concatenate(arg)
pred["nn_text"] = tr.text.values[nn]; pred["nn_label"] = tr.emotion.values[nn]
pred["nn_same_label"] = pred.nn_label == pred.emotion

bins = [0, 0.6, 0.8, 0.9, 1.01]; labels = ["< 0.6", "0.6–0.8", "0.8–0.9", "≥ 0.9"]
pred["sim_bin"] = pd.cut(pred.nn_sim, bins, labels=labels, right=False)
models = ["TF-IDF", "FastText", "SBERT"]
tab = pred.groupby("sim_bin", observed=True).apply(
    lambda g: pd.Series({"n": len(g), "share_%": 100 * len(g) / len(pred), "NN_same_label": g.nn_same_label.mean()}
                        | {m: (g[m] == g.emotion).mean() for m in models}))
tab.round(3).to_csv(RES / "accuracy_by_near_duplicate.csv")
print(tab.round(3))
summary = {"test_with_nn_sim_ge_0.9_%": round(100 * (pred.nn_sim >= 0.9).mean(), 1),
           "nearest_train_neighbour_same_label_%": round(100 * pred.nn_same_label.mean(), 1),
           "1-NN char-ngram classifier accuracy": round(pred.nn_same_label.mean(), 4)}
print(summary)
(RES / "near_duplicate_summary.json").write_text(json.dumps(summary, indent=1))
pred[pred.nn_sim >= 0.9].sample(8, random_state=1)[["text", "nn_text", "emotion", "nn_label", "nn_sim"]] \
    .to_csv(RES / "near_duplicate_examples.csv", index=False)

fig, ax = plt.subplots(figsize=(6.4, 2.6))
x = np.arange(len(tab)); w = 0.26
shades = {"TF-IDF": "#86b6ef", "FastText": "#2a78d6", "SBERT": "#184f95"}
for i, m in enumerate(models):
    ax.bar(x + (i - 1) * (w + 0.01), tab[m].values, w, color=shades[m], label=m)
ax.set_xticks(x, [f"{l}\n(n={int(n):,})" for l, n in zip(tab.index, tab.n)], fontsize=7.5)
ax.set_xlabel("Cosine similarity of test text to its nearest training text (char 3–4-grams)")
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
ax.legend(ncol=3, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.16))
fig.savefig(FIG / "fig7_near_duplicates.png"); plt.close(fig)
