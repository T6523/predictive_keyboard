"""Evaluate a num_bow_logreg*.pkl on dev's number-category rows (answer matches all-digit
regex -- every number in this dataset is a run of '1's, see eda/vocab_coverage_eda.ipynb).
Directly comparable to the ngram number-category accuracy reported in report.md.

Usage:
    python3 eval_on_dev.py --model num_bow_logreg_natural.pkl --out dev_number_predictions_natural.csv
"""
import argparse
import csv
import pickle
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
NUM_RE = re.compile(r"[0-9]+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="num_bow_logreg_balanced.pkl")
    ap.add_argument("--out", default="dev_number_predictions_balanced.csv")
    args = ap.parse_args()

    with open(HERE / args.model, "rb") as f:
        m = pickle.load(f)
    vec, clf = m["vectorizer"], m["model"]

    rows = []
    with open(DATA / "dev_set_final.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if NUM_RE.fullmatch(row["answer"]):
                rows.append(row)

    contexts = [r["context"] for r in rows]
    true_len = [len(r["answer"]) for r in rows]
    pred_len = clf.predict(vec.transform(contexts))

    correct = sum(p == t for p, t in zip(pred_len, true_len))
    print(f"dev number rows: {len(rows)}")
    print(f"accuracy: {correct}/{len(rows)} = {correct/len(rows):.4f}")

    out_path = HERE / args.out
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["context", "first letter", "answer", "true_length", "pred_length", "prediction", "correct"])
        for r, t, p in zip(rows, true_len, pred_len):
            pred_tok = "1" * int(p)
            w.writerow([r["context"], r["first letter"], r["answer"], t, p, pred_tok, int(p == t)])
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
