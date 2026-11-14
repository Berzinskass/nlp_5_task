# Task 5: Emotion classification of Lithuanian texts (EMO corpus)

Five-class emotion classification (neutral, joy, surprise, sadness, anger) of the transcripts of the
Lithuanian Emotional Speech Corpus (EMO) with TF-IDF, FastText and Sentence-BERT representations and
Logistic Regression. The report is in [`report/report.pdf`](report/report.pdf).

| Representation | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF (uni+bigrams) | 0.946 | 0.946 |
| FastText (trained on train set) | 0.635 | 0.631 |
| Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`) | 0.620 | 0.617 |

Note: ~48 % of test texts have a near-duplicate (transcription variant) in the training set, which
inflates the TF-IDF score; see Section 5 and Figure 7 of the report.

## Layout

- `data/emo_text_dataset.csv` – final dataset (text, label letter); `data/emo_text_processed.csv` adds lemmas/POS
- `src/` – code (`prepare_dataset.py` → `text_analysis.py` → `classify.py` → `near_duplicate_analysis.py`)
- `figures/`, `results/` – generated figures and tables
- `report/` – report source (`report.md`), build script and PDF

Folder letters: Z = neutral, D = joy (džiaugsmas), N = surprise (nustebimas), L = sadness (liūdesys), P = anger (pyktis).

## Reproduce

```bash
pip install -r requirements.txt
cp /path/to/metadata.csv data/metadata.csv          # raw corpus file (not committed)
python -c "from huggingface_hub import snapshot_download; snapshot_download('spacy/lt_core_news_md', local_dir='models/lt_core_news_md', ignore_patterns=['*.whl'])"
python src/prepare_dataset.py
python src/text_analysis.py
python src/classify.py
python src/near_duplicate_analysis.py
bash report/build.sh                                 # needs pandoc + Node Playwright/Chromium
```
