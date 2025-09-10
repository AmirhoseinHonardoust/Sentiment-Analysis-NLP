import argparse, os, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from joblib import dump
from wordcloud import WordCloud, STOPWORDS
from utils import preprocess

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def wordcloud_from_text(text, outpath):
    wc = WordCloud(width=1000, height=600, background_color="white", stopwords=STOPWORDS)
    img = wc.generate(text).to_image()
    img.save(outpath, format="PNG")

def top_tfidf_features(vectorizer, clf, k=25, outpath="outputs/top_features.txt", labels=None):
    feature_names = np.array(vectorizer.get_feature_names_out())
    lines = []
    if hasattr(clf, "coef_"):  # Logistic Regression (OvR)
        coefs = clf.coef_
        for idx, row in enumerate(coefs):
            top = np.argsort(row)[-k:][::-1]
            lab = labels[idx] if labels is not None else str(idx)
            lines.append(f"[{lab}] top +weights: " + ", ".join(feature_names[top]))
    elif hasattr(clf, "feature_importances_"):  # RandomForest
        importances = clf.feature_importances_
        top = np.argsort(importances)[-k:][::-1]
        lines.append("RandomForest top features: " + ", ".join(feature_names[top]))
    else:
        lines.append("Top features not available for this classifier.")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    df = load_dataset(args.input)
    df.dropna(subset=["text", "label"], inplace=True)
    df["text_clean"] = df["text"].astype(str).apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"], df["label"], test_size=args.test_size, stratify=df["label"], random_state=args.seed
    )

    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000)

    models = {
        "nb": MultinomialNB(),
        "logreg": LogisticRegression(max_iter=3000),
        "rf": RandomForestClassifier(n_estimators=300, random_state=args.seed)
    }

    metrics = {}
    best_name, best_score = None, -1.0
    best_vec, best_clf = None, None

    for name, clf in models.items():
        pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred, digits=3, zero_division=0)
        metrics[name] = {"f1_macro": float(f1), "report": report}
        if f1 > best_score:
            best_score = f1; best_name = name; best_vec = pipe.named_steps["tfidf"]; best_clf = pipe.named_steps["clf"]

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"models": metrics, "best_model": best_name}, f, indent=2)

    with open(os.path.join(args.outdir, "classification_report.txt"), "w") as f:
        f.write(f"Best model: {best_name}\n\n")
        f.write(metrics[best_name]["report"])

    # Confusion matrix
    y_pred_best = Pipeline([("tfidf", best_vec), ("clf", best_clf)]).fit(X_train, y_train).predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best, labels=["negative", "neutral", "positive"])
    fig, ax = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(cm, display_labels=["negative", "neutral", "positive"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (best model)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "confusion_matrix.png"), dpi=160)
    plt.close(fig)

    # Word clouds
    text_pos = " ".join(df[df["label"]=="positive"]["text_clean"].tolist())
    text_neg = " ".join(df[df["label"]=="negative"]["text_clean"].tolist())
    wordcloud_from_text(text_pos, os.path.join(args.outdir, "wordcloud_positive.png"))
    wordcloud_from_text(text_neg, os.path.join(args.outdir, "wordcloud_negative.png"))

    # Top features
    top_tfidf_features(best_vec, best_clf, k=30, outpath=os.path.join(args.outdir, "top_features.txt"), labels=best_clf.classes_ if hasattr(best_clf, "classes_") else None)

    # Save artifacts
    dump(best_vec, os.path.join(args.outdir, "vectorizer.joblib"))
    dump(best_clf, os.path.join(args.outdir, "best_model.joblib"))

    print("[OK] Training complete.") 
    print(f"Best model: {best_name} (F1-macro={best_score:.3f})") 
    print(f"Outputs saved to: {args.outdir}") 

if __name__ == "__main__":
    main()
