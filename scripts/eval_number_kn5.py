"""Eval the Gigaword+train number-only KN5 model against the current general n-gram's
number-category accuracy, on the same 2023-row set num/eval_on_dev.py used (apples-to-
apples with the 71.97% baseline already measured that way).

Candidates are "1","11",...,"1"*MAX_LEN (train's own observed max digit-token length is
8, see report.md) -- argmax of the model's log-prob given context, same shape as
KN5Scorer.score() in blend_ngram.py.

Usage:
    python3 eval_number_kn5.py --model ../weights/kn5_numbers.binary \
        --data ../num/dev_number_predictions_nb.csv
"""
import argparse
import csv

from blend_ngram import KN5Scorer

MAX_LEN = 8


def pick_length(scorer, context_tokens):
    best_len, best_score = 1, float("-inf")
    for length in range(1, MAX_LEN + 1):
        s = scorer.score(context_tokens, "1" * length)
        if s > best_score:
            best_len, best_score = length, s
    return best_len


def demo():
    """Self-check: pick_length argmax wiring, fake scorer, no model needed."""
    class FakeScorer:
        def score(self, ctx, cand):
            return -len(cand)  # shorter always wins
    assert pick_length(FakeScorer(), ["x"]) == 1
    print("demo ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../weights/kn5_numbers.binary")
    ap.add_argument("--data", default="../num/dev_number_predictions_nb.csv")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    scorer = KN5Scorer(args.model)
    rows = list(csv.DictReader(open(args.data, encoding="utf-8")))

    correct = 0
    for r in rows:
        ctx = r["context"].split()
        pred_len = pick_length(scorer, ctx)
        pred = "1" * pred_len
        correct += pred == r["answer"]

    print(f"gigaword+train number KN5: {correct}/{len(rows)} = {correct/len(rows)*100:.2f}%")


if __name__ == "__main__":
    main()
