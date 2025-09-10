import argparse, numpy as np, pandas as pd, random

POSITIVE_SEEDS = [
    "Amazing quality and fast delivery! Totally satisfied.",
    "Great value for the price. Will buy again.",
    "Customer support was helpful and polite.",
    "I love this product. Highly recommended.",
    "Exceeded my expectations in every way."
]
NEGATIVE_SEEDS = [
    "Terrible experience. The item arrived broken.",
    "Very poor quality and slow shipping.",
    "Not worth the money. I want a refund.",
    "Support was unhelpful and rude.",
    "Completely disappointed. Do not recommend."
]
NEUTRAL_SEEDS = [
    "Works as expected. Nothing special.",
    "Okay product, decent for everyday use.",
    "Average experience overall.",
    "The item matches the description.",
    "It's fine, does the job."
]

def synthesize(n, seed=42):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["positive", "neutral", "negative"], size=n, p=[0.45, 0.25, 0.30])
    rows = []
    for i, lab in enumerate(labels, start=1):
        if lab == "positive":
            base = random.choice(POSITIVE_SEEDS)
            extras = ["fast", "reliable", "excellent", "love", "five stars", "great"]
        elif lab == "negative":
            base = random.choice(NEGATIVE_SEEDS)
            extras = ["late", "broken", "bad", "refund", "waste", "one star"]
        else:
            base = random.choice(NEUTRAL_SEEDS)
            extras = ["okay", "average", "standard", "fine", "works", "normal"]
        words = base.split()
        words += rng.choice(extras, size=rng.integers(0, 6), replace=True).tolist()
        if rng.random() < 0.1: words.append("http://example.com")
        if rng.random() < 0.1: words.append("👍")
        if rng.random() < 0.1: words.append("#deal")
        text = " ".join(words)
        rows.append([i, text, lab])
    return pd.DataFrame(rows, columns=["review_id", "text", "label"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/reviews.csv")
    args = ap.parse_args()

    df = synthesize(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out} with {len(df):,} rows")

if __name__ == "__main__":
    main()
