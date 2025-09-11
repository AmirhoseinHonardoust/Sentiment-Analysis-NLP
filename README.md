# Sentiment Analysis (NLP)

Customer review sentiment analysis with Python and NLP.  
The project uses a **synthetic review dataset** (positive, neutral, negative), applies text preprocessing (cleaning, tokenization, stopwords removal, lemmatization), converts text to **TF-IDF features**, and trains classifiers (Naive Bayes, Logistic Regression, Random Forest).  
The best model is selected based on macro F1-score, and results are visualized with confusion matrix, word clouds, and top TF-IDF features.

---

## Features
- Generate synthetic review dataset
- Text preprocessing:
  - lowercasing, URL & punctuation removal
  - stopwords filtering
  - lemmatization
- TF-IDF vectorization (unigrams + bigrams)
- Models: Multinomial Naive Bayes, Logistic Regression, Random Forest
- Evaluation: accuracy, precision, recall, F1-score
- Visuals: confusion matrix, word clouds, top features per class
- Saved artifacts: best model + vectorizer (`joblib`), metrics JSON

---

## Project Structure
```
sentiment-analysis-nlp/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ data/
│  └─ generate_reviews.py
├─ src/
│  ├─ train_nlp.py
│  └─ utils.py
└─ outputs/
   └─ figures & reports (auto-created)
```

---

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Generate Synthetic Reviews
```bash
python data/generate_reviews.py --n 8000 --seed 42 --out data/reviews.csv
```

---

## Train & Evaluate
```bash
python src/train_nlp.py --input data/reviews.csv --outdir outputs --test-size 0.2 --seed 42
```

**Outputs**
- `metrics.json` – per-model scores & best model
- `classification_report.txt`
- `confusion_matrix.png`
- `wordcloud_positive.png`, `wordcloud_negative.png`
- `top_features.txt`
- `best_model.joblib`, `vectorizer.joblib`

---

## Example Results

### Confusion Matrix
Best model performance across classes:  
<img width="960" height="960" alt="confusion_matrix" src="https://github.com/user-attachments/assets/feac74b0-cdf1-467d-add0-97535d9ab8b9" />

---

### Word Cloud (Positive Reviews)
<img width="1000" height="600" alt="wordcloud_positive" src="https://github.com/user-attachments/assets/698eeb83-3975-46b9-8173-3bb5228be4cb" />

---

### Word Cloud (Negative Reviews)
<img width="1000" height="600" alt="wordcloud_negative" src="https://github.com/user-attachments/assets/081da50c-c54d-4ab2-915e-b61799a821ba" />

---

### Top Features
File: `outputs/top_features.txt`  
Shows top discriminative words/phrases learned by the classifier for each class.

---

## Data Schema
| column     | description                            |
|------------|----------------------------------------|
| review_id  | unique id                              |
| text       | raw review text                        |
| label      | sentiment {negative, neutral, positive} |
