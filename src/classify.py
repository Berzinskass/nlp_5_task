"""Parts 3-4: three text representations + Logistic Regression emotion classifiers.

1. TF-IDF (word uni+bigrams)                      - frequency-based
2. FastText trained on the training set (gensim)  - static word embeddings, mean-pooled
3. paraphrase-multilingual-MiniLM-L12-v2 (SBERT)  - contextual sentence embeddings
All data-dependent fitting (vocabulary, IDF, FastText training, scaler, LR) uses the training set only.
"""
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import FastText
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import COLORS, DATA, FIG, ORDER, RES, ROOT, SEED, set_style

set_style()
CACHE = ROOT / "models" / "cache"; CACHE.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "emo_text_processed.csv")
y = df["emotion"].values
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
# validation slice (from training data only) used to choose C
idx_fit, idx_val = train_test_split(idx_tr, test_size=0.1, stratify=y[idx_tr], random_state=SEED)
split = pd.Series("train", index=df.index); split.iloc[idx_te] = "test"
df.assign(split=split)[["text", "emotion", "split"]].to_csv(DATA / "train_test_split.csv", index=False)
print("train", len(idx_tr), "test", len(idx_te))
texts = df["text"].tolist()
tok = [t.split() for t in texts]

info, X = {}, {}

# ------------------------------------------------------------------ 1. TF-IDF
t0 = time.time()
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
tfidf.fit([texts[i] for i in idx_tr])
X["TF-IDF"] = tfidf.transform(texts).tocsr()
info["TF-IDF"] = {"dim": X["TF-IDF"].shape[1], "fit_s": time.time() - t0}

# ------------------------------------------------------------------ 2. FastText (trained on train only)
t0 = time.time()
ft = FastText(vector_size=300, window=5, min_count=2, sg=1, epochs=10, min_n=3, max_n=6, workers=4, seed=SEED)
ft.build_vocab([tok[i] for i in idx_tr])
ft.train([tok[i] for i in idx_tr], total_examples=len(idx_tr), epochs=ft.epochs)
X["FastText"] = np.vstack([ft.wv.get_mean_vector(t) if t else np.zeros(300) for t in tok]).astype(np.float32)
info["FastText"] = {"dim": 300, "fit_s": time.time() - t0, "vocab": len(ft.wv)}
neigh = {w: [(n, round(float(s), 3)) for n, s in ft.wv.most_similar(w, topn=6)]
         for w in ["džiaugsmas", "liūdna", "pyktis", "neįtikėtina", "kvailys", "miestas"] if w in ft.wv}
(RES / "fasttext_neighbours.json").write_text(json.dumps(neigh, ensure_ascii=False, indent=1))
print(neigh)

# ------------------------------------------------------------------ 3. Sentence-BERT (pretrained, frozen)
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
cache = CACHE / "sbert_minilm.npy"
t0 = time.time()
if cache.exists():
    X["SBERT"] = np.load(cache)
else:
    sbert = SentenceTransformer(MODEL, device="cpu")
    X["SBERT"] = sbert.encode(texts, batch_size=128, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    np.save(cache, X["SBERT"])
info["SBERT"] = {"dim": X["SBERT"].shape[1], "fit_s": time.time() - t0, "model": MODEL}

# ------------------------------------------------------------------ representation properties
for name, M in X.items():
    nnz = M.nnz if sparse.issparse(M) else int(np.count_nonzero(M))
    info[name]["sparsity_%"] = round(100 * (1 - nnz / (M.shape[0] * M.shape[1])), 3)
    info[name]["mean_nonzero_per_text"] = round(nnz / M.shape[0], 1)
print(json.dumps(info, indent=1))

# ------------------------------------------------------------------ classification
def make_lr(C):
    return LogisticRegression(C=C, max_iter=3000, solver="lbfgs", random_state=SEED)

GRID = {"TF-IDF": [1, 3, 10, 30], "FastText": [0.01, 0.1, 1, 10], "SBERT": [0.01, 0.1, 1, 10]}
results, preds, cms = [], {}, {}
for name, M in X.items():
    if not sparse.issparse(M):  # standardise dense embeddings (scaler fitted on training data)
        sc = StandardScaler().fit(M[idx_fit]); Mf, Mv = sc.transform(M[idx_fit]), sc.transform(M[idx_val])
    else:
        Mf, Mv = M[idx_fit], M[idx_val]
    val = {}
    for C in GRID[name]:
        val[C] = precision_recall_fscore_support(y[idx_val], make_lr(C).fit(Mf, y[idx_fit]).predict(Mv), average="macro")[2]
    best_C = max(val, key=val.get)
    if not sparse.issparse(M):
        sc = StandardScaler().fit(M[idx_tr]); Mtr, Mte = sc.transform(M[idx_tr]), sc.transform(M[idx_te])
    else:
        Mtr, Mte = M[idx_tr], M[idx_te]
    t0 = time.time(); clf = make_lr(best_C).fit(Mtr, y[idx_tr]); train_s = time.time() - t0
    p = clf.predict(Mte); preds[name] = p
    P, R, F, _ = precision_recall_fscore_support(y[idx_te], p, average="macro")
    Pw, Rw, Fw, _ = precision_recall_fscore_support(y[idx_te], p, average="weighted")
    results.append({"representation": name, "dim": info[name]["dim"], "C": best_C, "val_macro_F1": round(val[best_C], 4),
                    "accuracy": accuracy_score(y[idx_te], p), "precision_macro": P, "recall_macro": R, "F1_macro": F,
                    "F1_weighted": Fw, "train_s": round(train_s, 1)})
    rep = classification_report(y[idx_te], p, labels=ORDER, output_dict=True)
    pd.DataFrame(rep).T.round(4).to_csv(RES / f"report_{name}.csv")
    cms[name] = confusion_matrix(y[idx_te], p, labels=ORDER)
    print(name, "val", {k: round(v, 4) for k, v in val.items()}, "->", results[-1])

res = pd.DataFrame(results).round(4)
res.to_csv(RES / "classification_results.csv", index=False)
(RES / "representation_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
per_class = pd.concat({n: pd.read_csv(RES / f"report_{n}.csv", index_col=0).loc[ORDER, ["precision", "recall", "f1-score"]]
                       for n in X}, axis=1)
per_class.round(3).to_csv(RES / "per_class_metrics.csv")
print(res.to_string()); print(per_class.round(3))

# ------------------------------------------------------------------ confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, (name, cm) in zip(axes, cms.items()):
    cmn = cm / cm.sum(axis=1, keepdims=True) * 100
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    for (i, j), v in np.ndenumerate(cmn):
        ax.text(j, i, f"{v:.1f}\n({cm[i, j]:,})", ha="center", va="center", fontsize=6.5,
                color="white" if v > 55 else "#0b0b0b")
    ax.set_xticks(range(5), ORDER, rotation=30, fontsize=7.5); ax.set_yticks(range(5), ORDER, fontsize=7.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True" if name == "TF-IDF" else "")
    acc = res.set_index("representation").loc[name, "accuracy"]
    ax.set_title(f"{name} (acc. {acc:.3f})"); ax.grid(False)
fig.tight_layout(); fig.savefig(FIG / "fig6_confusion_matrices.png"); plt.close(fig)

# ------------------------------------------------------------------ error analysis
te = df.iloc[idx_te][["text", "emotion"]].reset_index(drop=True)
for name in X: te[name] = preds[name]
te["n_correct"] = sum((te[n] == te.emotion).astype(int) for n in X)
te.to_csv(RES / "test_predictions.csv", index=False)
pairs = []
for name, cm in cms.items():
    off = [(ORDER[i], ORDER[j], int(cm[i, j]), round(100 * cm[i, j] / cm[i].sum(), 1))
           for i in range(5) for j in range(5) if i != j]
    for t, pr, c, pct in sorted(off, key=lambda r: -r[2])[:5]:
        pairs.append({"representation": name, "true": t, "predicted": pr, "count": c, "pct_of_true": pct})
pd.DataFrame(pairs).to_csv(RES / "top_confusions.csv", index=False)
print(pd.DataFrame(pairs).to_string())
print("all three wrong: %.1f%%, all right: %.1f%%" % (100 * (te.n_correct == 0).mean(), 100 * (te.n_correct == 3).mean()))
te["len"] = te.text.str.split().str.len()
acc_len = te.assign(bin=pd.cut(te.len, [0, 3, 6, 10, 15, 100], labels=["1-3", "4-6", "7-10", "11-15", "16+"]))
acc_len = acc_len.groupby("bin", observed=True).apply(lambda g: pd.Series({n: (g[n] == g.emotion).mean() for n in X} | {"n": len(g)}))
acc_len.round(3).to_csv(RES / "accuracy_by_length.csv"); print(acc_len.round(3))
(te[te.n_correct == 0].sample(25, random_state=SEED)).to_csv(RES / "hard_examples.csv", index=False)

# ------------------------------------------------------------------ near-duplicate check (train vs test)
cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2).fit([texts[i] for i in idx_tr])
A, B = cv.transform([texts[i] for i in idx_te]), cv.transform([texts[i] for i in idx_tr]).T.tocsc()
best = np.concatenate([(A[s:s + 2000] @ B).max(axis=1).toarray().ravel() for s in range(0, A.shape[0], 2000)])
nd = {f">= {t}": round(float((best >= t).mean() * 100), 2) for t in (0.8, 0.9, 0.95)}
(RES / "near_duplicates.json").write_text(json.dumps(nd, indent=1)); print("near-duplicate test texts (%):", nd)
print("done")
