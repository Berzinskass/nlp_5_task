
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from spacy.lang.lt.stop_words import STOP_WORDS
from wordcloud import WordCloud

# spaCy LT list + auxiliary/modal verbs that the list misses and that top every class
EXTRA_STOP = {"būti", "galėti", "turėti", "tapti"}
STOP = set(STOP_WORDS) | EXTRA_STOP

from common import COLORS, DATA, FIG, LABEL_NAMES, ORDER, RES, SPACY_MODEL, set_style

set_style()
df = pd.read_csv(DATA / "emo_text_dataset.csv")
df["emotion"] = df["label"].map(LABEL_NAMES)



counts = df["emotion"].value_counts().reindex(ORDER)
stats = json.loads((RES / "dataset_stats.json").read_text())
before = pd.Series(stats["samples_before_dedup_per_class"]).rename(index=LABEL_NAMES).reindex(ORDER)
fig, ax = plt.subplots(figsize=(6.2, 2.6))
x = np.arange(len(ORDER)); w = 0.38
ax.bar(x - w / 2 - 0.01, before.values, w, color=[COLORS[e] for e in ORDER], alpha=0.35, label="before duplicate removal")
ax.bar(x + w / 2 + 0.01, counts.values, w, color=[COLORS[e] for e in ORDER], label="after duplicate removal")
for xi, v in zip(x, counts.values):
    ax.text(xi + w / 2, v + 900, f"{v:,}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x, ORDER); ax.set_ylabel("Number of texts"); ax.grid(axis="x", visible=False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v/1000)}k"))
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#898781", alpha=0.35, label="before duplicate removal"),
                   Patch(color="#898781", label="after duplicate removal")],
          loc="upper left", fontsize=7.5, ncol=2, bbox_to_anchor=(0, 1.15))
fig.savefig(FIG / "fig1_class_distribution.png"); plt.close(fig)



df["n_words"] = df["text"].str.split().str.len()
len_stats = df.groupby("emotion")["n_words"].describe().reindex(ORDER)
len_stats["median"] = df.groupby("emotion")["n_words"].median().reindex(ORDER)
len_stats.round(2).to_csv(RES / "length_stats.csv")
print(len_stats.round(2))

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
data = [df.loc[df.emotion == e, "n_words"].values for e in ORDER]
bp = ax.boxplot(data, tick_labels=ORDER, showfliers=False, widths=0.55, patch_artist=True,
                medianprops={"color": "#0b0b0b", "lw": 1.2}, whiskerprops={"color": "#898781"},
                capprops={"color": "#898781"})
for patch, e in zip(bp["boxes"], ORDER):
    patch.set_facecolor(COLORS[e]); patch.set_edgecolor("white"); patch.set_alpha(0.85)
for i, e in enumerate(ORDER, 1):
    ax.plot(i, len_stats.loc[e, "mean"], marker="D", ms=4, color="white", mec="#0b0b0b", mew=0.8)
ax.set_ylabel("Text length (words)"); ax.set_title("(a) Box plots (outliers hidden; ◇ = mean)")
ax.grid(axis="x", visible=False)
ax = axes[1]
bins = np.arange(1, 31)
for e in ORDER:
    h, _ = np.histogram(df.loc[df.emotion == e, "n_words"], bins=bins, density=True)
    ax.plot(bins[:-1], h * 100, color=COLORS[e], lw=1.6, label=e)
ax.set_xlabel("Text length (words)"); ax.set_ylabel("Share of class texts (%)")
ax.set_title("(b) Length distributions (1–29 words)"); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(FIG / "fig2_text_length.png"); plt.close(fig)



nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
lemmas, pos_tags = [], []
for doc in nlp.pipe(df["text"].tolist(), batch_size=1000, n_process=4):
    pos_tags.append([t.pos_ for t in doc])
    lemmas.append([t.lemma_.lower() for t in doc
                   if t.is_alpha and len(t) > 1
                   and t.text.lower() not in STOP and t.lemma_.lower() not in STOP])
df["lemmas"] = [" ".join(l) for l in lemmas]
df["pos"] = [" ".join(p) for p in pos_tags]
df[["text", "label", "emotion", "n_words", "lemmas", "pos"]].to_csv(DATA / "emo_text_processed.csv", index=False)



freq = {e: Counter(w for s in df.loc[df.emotion == e, "lemmas"] for w in s.split()) for e in ORDER}
TOPN = 15
top = {e: freq[e].most_common(TOPN) for e in ORDER}
rows = []
for e in ORDER:
    n_tok = sum(freq[e].values())
    for r, (w, c) in enumerate(top[e], 1):
        rows.append({"emotion": e, "rank": r, "lemma": w, "count": c, "per_1000": round(1000 * c / n_tok, 2)})
pd.DataFrame(rows).to_csv(RES / "top_lemmas.csv", index=False)
vocab_stats = pd.DataFrame({
    "content_tokens": {e: sum(freq[e].values()) for e in ORDER},
    "unique_lemmas": {e: len(freq[e]) for e in ORDER},
}).reindex(ORDER)
vocab_stats["type_token_ratio"] = (vocab_stats.unique_lemmas / vocab_stats.content_tokens).round(3)
vocab_stats.to_csv(RES / "vocab_stats.csv")
print(vocab_stats)

# overlap of top-100 lemmas between classes (Jaccard)
top100 = {e: {w for w, _ in freq[e].most_common(100)} for e in ORDER}
jac = pd.DataFrame([[len(top100[a] & top100[b]) / len(top100[a] | top100[b]) for b in ORDER] for a in ORDER],
                   index=ORDER, columns=ORDER).round(3)
jac.to_csv(RES / "top100_jaccard.csv")
print(jac)

# distinctive lemmas: log-odds ratio with informative Dirichlet prior (Monroe et al., 2008)
total = sum(freq.values(), Counter()); a0 = sum(total.values())
distinct = {}
for e in ORDER:
    rest = total - freq[e]; n_i, n_j = sum(freq[e].values()), sum(rest.values())
    z = {}
    for w, c in freq[e].items():
        if total[w] < 30: continue
        aw = total[w] / a0 * 1000
        d = np.log((c + aw) / (n_i + 1000 - c - aw)) - np.log((rest[w] + aw) / (n_j + 1000 - rest[w] - aw))
        z[w] = d / np.sqrt(1 / (c + aw) + 1 / (rest[w] + aw))
    distinct[e] = [w for w, _ in sorted(z.items(), key=lambda kv: -kv[1])[:15]]
pd.DataFrame(distinct).to_csv(RES / "distinctive_lemmas.csv", index=False)

fig, axes = plt.subplots(1, 5, figsize=(10, 3.6))
for ax, e in zip(axes, ORDER):
    words = [w for w, _ in top[e]][::-1]
    vals = [1000 * c / sum(freq[e].values()) for _, c in top[e]][::-1]
    ax.barh(words, vals, color=COLORS[e], height=0.7)
    ax.set_title(e); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=7.5, colors="#0b0b0b")
    ax.set_xlabel("per 1,000 tokens", fontsize=7.5)
fig.tight_layout(); fig.savefig(FIG / "fig3_top_lemmas.png"); plt.close(fig)

fig, axes = plt.subplots(1, 5, figsize=(12, 2.4))
for ax, e in zip(axes.flat, ORDER):
    wc = WordCloud(width=600, height=480, background_color="white", max_words=80, random_state=0,
                   color_func=lambda *a, c=COLORS[e], **k: c, prefer_horizontal=0.9, collocations=False)
    ax.imshow(wc.generate_from_frequencies(freq[e]), interpolation="bilinear")
    ax.set_title(e, fontsize=11); ax.axis("off")
fig.tight_layout(w_pad=0.6); fig.savefig(FIG / "fig4_wordclouds.png"); plt.close(fig)



pos_counts = pd.DataFrame({e: Counter(p for s in df.loc[df.emotion == e, "pos"] for p in s.split()) for e in ORDER}).fillna(0)
pos_share = (pos_counts / pos_counts.sum() * 100)
pos_share = pos_share.loc[pos_share.mean(axis=1).sort_values(ascending=False).index]
pos_share = pos_share[pos_share.mean(axis=1) >= 0.3]
pos_share.round(2).to_csv(RES / "pos_distribution.csv")
print(pos_share.round(2))
pos_per_text = pd.DataFrame({e: (pos_counts[e] / (df.emotion == e).sum()) for e in ORDER}).loc[pos_share.index]
pos_per_text.round(3).to_csv(RES / "pos_per_text.csv")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
tags = list(pos_share.index); x = np.arange(len(tags)); w = 0.16
for i, e in enumerate(ORDER):
    ax.bar(x + (i - 2) * (w + 0.008), pos_share[e].values, w, color=COLORS[e], label=e)
ax.set_xticks(x, tags, rotation=45, fontsize=7.5); ax.set_ylabel("Share of tokens (%)")
ax.set_title("(a) POS tag distribution per class"); ax.legend(fontsize=7.5, ncol=5, loc="upper right")
ax.grid(axis="x", visible=False)
ax = axes[1]
dev = pos_share.sub(pos_share.mean(axis=1), axis=0).head(12)
lim = np.abs(dev.values).max()
im = ax.imshow(dev.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(range(5), ORDER, rotation=30, fontsize=7.5); ax.set_yticks(range(len(dev)), dev.index, fontsize=7.5)
for (i, j), v in np.ndenumerate(dev.values):
    ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > lim * 0.6 else "#0b0b0b")
ax.grid(False); ax.set_title("(b) Deviation from mean share (p.p.)")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout(); fig.savefig(FIG / "fig5_pos.png"); plt.close(fig)
print("done")
