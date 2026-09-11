"""Multinomial Naive Bayes on raw bag-of-word counts (not TF-IDF) for number digit-length.

Why NB over the logreg attempt: TF-IDF's idf term downweights exactly the frequent words
("about", "nearly", "$", "percent", ...) that carry the length-cue and base-rate signal --
counterproductive here, the opposite of a normal text-classification corpus where frequent
words are uninformative stopwords. Raw counts + NB is a closed-form fit (seconds, not
minutes, unlike logreg's iterative optimizer) so it can afford the full 2.6M-row natural
distribution instead of a 600k subsample.

Usage:
    python3 train_nb.py --data train_number_contexts_natural_full.csv --out num_nb.pkl
"""
import argparse
import pickle
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_number_contexts_natural_full.csv")
    ap.add_argument("--out", default="num_nb.pkl")
    args = ap.parse_args()

    df = pd.read_csv(HERE / args.data)
    X_train, X_val, y_train, y_val = train_test_split(
        df["context"], df["length"], test_size=0.1, random_state=0, stratify=df["length"]
    )

    t0 = time.time()
    vec = CountVectorizer(max_features=50_000)
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)
    print(f"vectorized {Xtr.shape[0]} train / {Xva.shape[0]} val rows, {Xtr.shape[1]} features, {time.time()-t0:.1f}s")

    t0 = time.time()
    clf = MultinomialNB()
    clf.fit(Xtr, y_train)
    print(f"trained in {time.time()-t0:.1f}s")

    pred = clf.predict(Xva)
    acc = accuracy_score(y_val, pred)
    print(f"\ninternal held-out val accuracy: {acc:.4f}")
    print(classification_report(y_val, pred, zero_division=0))

    with open(HERE / args.out, "wb") as f:
        pickle.dump({"vectorizer": vec, "model": clf}, f)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
