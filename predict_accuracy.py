#!/usr/bin/env python3
"""Next-word prediction accuracy for KenLM models trained by pipeline.py.

Not part of the log-prob scoring pipeline.py does -- that scores whole context sentences.
This answers a different question: given a context + required first letter, which candidate
word does the model rank highest, and does that match the csv's `answer` column.

For each row: prime a KenLM state on the context tokens (once), then score every vocab
candidate starting with the letter as one incremental step from that state (kenlm.BaseScore),
argmax, compare to answer. Candidate pool is weights/vocab.txt (same file eval_ngram.py uses --
built by scripts/build_vocab.py from the alnum-only corpus, so it already matches the answer
column's format; matches Model A's vocab, which Model B was masked to per the pipeline spec).

Usage:
    python3 predict_accuracy.py --model weights/run_.../model_a.klm --data data/devv_eval.csv
    python3 predict_accuracy.py --model weights/run_.../model_b.klm --data data/devv_test.csv --limit 2000
"""
import argparse
import csv
import time
from pathlib import Path

import kenlm

ROOT = Path(__file__).resolve().parent


def load_vocab_by_letter(vocab_path):
    by_letter = {}
    with open(vocab_path, encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                by_letter.setdefault(w[0], []).append(w)
    return by_letter


def predict(model, context_tokens, letter, vocab_by_letter, state_a, state_b):
    """Prime state on context, score every same-letter candidate as one incremental step."""
    model.BeginSentenceWrite(state_a)
    for tok in context_tokens:
        model.BaseScore(state_a, tok, state_b)
        state_a, state_b = state_b, state_a
    candidates = vocab_by_letter.get(letter.lower(), ())
    best_word, best_score = None, float("-inf")
    for word in candidates:
        score = model.BaseScore(state_a, word, state_b)
        if score > best_score:
            best_word, best_score = word, score
    return best_word


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path, help="path to a .klm model")
    ap.add_argument("--data", required=True, type=Path, help="csv with context, first letter, answer columns")
    ap.add_argument("--vocab", default=ROOT / "weights" / "vocab.txt", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model = kenlm.Model(str(args.model))
    vocab_by_letter = load_vocab_by_letter(args.vocab)
    n_candidates = sum(len(v) for v in vocab_by_letter.values())
    print(f"loaded {args.model.name}, vocab: {n_candidates} words / {len(vocab_by_letter)} letters")

    state_a, state_b = kenlm.State(), kenlm.State()
    correct = no_candidates = 0
    n = 0
    t0 = time.time()
    with open(args.data, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.limit and n >= args.limit:
                break
            n += 1
            pred = predict(model, row["context"].split(), row["first letter"],
                            vocab_by_letter, state_a, state_b)
            if pred is None:
                no_candidates += 1
            elif pred == row["answer"]:
                correct += 1

    print(f"rows: {n}")
    print(f"accuracy: {correct}/{n} = {correct/n:.4f}")
    print(f"no candidates for letter: {no_candidates}")
    print(f"eval time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
