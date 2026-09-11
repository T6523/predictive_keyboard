"""TF-IDF + logistic regression classifier for number digit-length, trained on a
train_number_contexts*.csv file (see extract_train_numbers.py). Saves the fitted
vectorizer + model to a .pkl.

Usage:
    python3 train_bow_logreg.py --data train_number_contexts_natural.csv \
        --out num_bow_logreg_natural.pkl
"""
import argparse
import pickle
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_number_contexts_balanced.csv")
    ap.add_argument("--out", default="num_bow_logreg_balanced.pkl")
    args = ap.parse_args()

    df = pd.read_csv(HERE / args.data)
    X_train, X_val, y_train, y_val = train_test_split(
        df["context"], df["length"], test_size=0.1, random_state=0, stratify=df["length"]
    )

    t0 = time.time()
    vec = TfidfVectorizer(max_features=50_000)
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)
    print(f"vectorized {Xtr.shape[0]} train / {Xva.shape[0]} val rows, {Xtr.shape[1]} features, {time.time()-t0:.1f}s")

    t0 = time.time()
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(Xtr, y_train)
    print(f"trained in {time.time()-t0:.1f}s")

    pred = clf.predict(Xva)
    acc = accuracy_score(y_val, pred)
    print(f"\ninternal held-out val accuracy (10% of train_number_contexts.csv): {acc:.4f}")
    print(classification_report(y_val, pred, zero_division=0))

    with open(HERE / args.out, "wb") as f:
        pickle.dump({"vectorizer": vec, "model": clf}, f)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
