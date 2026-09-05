#!/usr/bin/env python3
"""Next-word prediction accuracy for KenLM models trained by pipeline.py.

Not part of the log-prob scoring pipeline.py does -- that scores whole context sentences.
This answers a different question: given a context + required first letter, which candidate
word does the model rank highest, and does that match the csv's `answer` column.

For each row: prime a KenLM state on the context tokens (once), then score every vocab
candidate starting with the letter as one incremental step from that state (kenlm.BaseFullScore).
Rank by (ngram_length desc, log_prob desc): deepest actually-observed n-gram order wins first,
smoothed log-prob only breaks ties within the same order -- raw BaseScore argmax overweights
Kneser-Ney smoothing and picks common words the context never actually preceded (see TODO.md).
Candidate pool is weights/vocab.txt (same file eval_ngram.py uses --
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

from scripts.symbol_predict import is_symbol_letter, predict_symbol

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
    out_state = kenlm.State()
    best_word, best_key = None, (-1, float("-inf"))
    for word in candidates:
        full = model.BaseFullScore(state_a, word, out_state)
        key = (full.ngram_length, full.log_prob)
        if key > best_key:
            best_word, best_key = word, key
    return best_word


def _prime(model, context_tokens, s1, s2):
    """Returns whichever of s1/s2 ends up holding the final context state."""
    model.BeginSentenceWrite(s1)
    for tok in context_tokens:
        model.BaseScore(s1, tok, s2)
        s1, s2 = s2, s1
    return s1


def predict_interp(model_a, model_b, context_tokens, letter, vocab_by_letter, lam, sa1, sa2, sb1, sb2):
    """Same idea as predict(), but argmax over lam*P_a + (1-lam)*P_b per candidate
    (probability-space interpolation, matches interpolate.py's perplexity math)."""
    sa = _prime(model_a, context_tokens, sa1, sa2)
    sb = _prime(model_b, context_tokens, sb1, sb2)
    candidates = vocab_by_letter.get(letter.lower(), ())
    out_a, out_b = kenlm.State(), kenlm.State()
    best_word, best_prob = None, -1.0
    for word in candidates:
        log_a = model_a.BaseFullScore(sa, word, out_a).log_prob
        log_b = model_b.BaseFullScore(sb, word, out_b).log_prob
        prob = lam * (10 ** log_a) + (1 - lam) * (10 ** log_b)
        if prob > best_prob:
            best_word, best_prob = word, prob
    return best_word


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path, help="path to a .klm model (model A if --model-b given)")
    ap.add_argument("--model-b", type=Path, help="second .klm model -- enables interpolated prediction")
    ap.add_argument("--lam", type=float, default=0.71, help="interpolation weight for --model (perplexity-tuned default)")
    ap.add_argument("--data", required=True, type=Path, help="csv with context, first letter, answer columns")
    ap.add_argument("--vocab", default=ROOT / "weights" / "vocab.txt", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model = kenlm.Model(str(args.model))
    model_b = kenlm.Model(str(args.model_b)) if args.model_b else None
    vocab_by_letter = load_vocab_by_letter(args.vocab)
    n_candidates = sum(len(v) for v in vocab_by_letter.values())
    tag = f"{args.model.name}+{args.model_b.name} (lam={args.lam})" if model_b else args.model.name
    print(f"loaded {tag}, vocab: {n_candidates} words / {len(vocab_by_letter)} letters")

    state_a, state_b = kenlm.State(), kenlm.State()
    sa1, sa2, sb1, sb2 = kenlm.State(), kenlm.State(), kenlm.State(), kenlm.State()
    correct = no_candidates = 0
    n_symbol = correct_symbol = n_alnum = correct_alnum = 0
    n = 0
    t0 = time.time()
    with open(args.data, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.limit and n >= args.limit:
                break
            n += 1
            letter = row["first letter"]
            if is_symbol_letter(letter):
                n_symbol += 1
                pred = predict_symbol(letter)
            else:
                n_alnum += 1
                tokens = row["context"].split()
                if model_b:
                    pred = predict_interp(model, model_b, tokens, letter,
                                           vocab_by_letter, args.lam, sa1, sa2, sb1, sb2)
                else:
                    pred = predict(model, tokens, letter, vocab_by_letter, state_a, state_b)
            if pred is None:
                no_candidates += 1
            elif pred == row["answer"]:
                correct += 1
                if is_symbol_letter(letter):
                    correct_symbol += 1
                else:
                    correct_alnum += 1

    print(f"rows: {n}")
    print(f"accuracy: {correct}/{n} = {correct/n:.4f}")
    print(f"  alnum route:  {correct_alnum}/{n_alnum} = {correct_alnum/n_alnum:.4f}" if n_alnum else "  alnum route: n/a")
    print(f"  symbol route: {correct_symbol}/{n_symbol} = {correct_symbol/n_symbol:.4f}" if n_symbol else "  symbol route: n/a")
    print(f"no candidates for letter: {no_candidates}")
    print(f"eval time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
