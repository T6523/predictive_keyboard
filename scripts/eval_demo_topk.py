"""Evaluate the browser-demo table's top-K accuracy on dev_set_final.csv, using the exact
same category routing as infer_ngram.py (symbol -> trivial letter copy, number/word -> ngram
lookup) but counting a hit whenever the true answer is ANYWHERE in the top-K, not just #1.

Builds the tables in-memory with build_topk_table (same function export_demo_model.py uses to
produce the shipped bytes) at whatever --min-count/--max-prefix-len/--top-k/--strict the demo
was actually configured with, then walks the same order-(n-1)->...->1 backoff predict_word()
uses, so this measures exactly what the shipped browser table would answer.

Usage:
    python3 eval_demo_topk.py --model ../weights/ngram_4_a.bin --dev ../data/dev_set_final.csv \
        --min-count 10 --max-prefix-len 1 --top-k 3 --strict
"""
import argparse
import csv
import pickle
import re
from collections import defaultdict

from export_demo_model import build_topk_table

CAT_NUMBER = re.compile(r"[0-9]+")
CAT_WORD = re.compile(r"(?=.*[a-z])[a-z']+", re.IGNORECASE)


def categorize(tok):
    if CAT_NUMBER.fullmatch(tok):
        return "number"
    if CAT_WORD.fullmatch(tok):
        return "word"
    return "symbol"


def load_model(path):
    with open(path, "rb") as f:
        m = pickle.load(f)
    return m["n"], m["counts"], m["vocab"], m["id_to_tok"]


def make_predictor(n, tables, vocab, id_to_tok):
    """Same backoff order as predict_word/topk_by_letter in infer_ngram.py: try the longest
    available context first (up to n-1 tokens), shrink on a miss or an OOV token, down to the
    order-0 (unigram, no context) table as the final fallback."""
    def predict_topk(context_tokens, letter):
        ids = [vocab.get(t) for t in context_tokens[-(n - 1):]]
        for j in range(len(ids), 0, -1):
            ctx_ids = tuple(ids[-j:])
            if None in ctx_ids:
                continue
            by_prefix = tables[j].get(ctx_ids)
            if by_prefix:
                wids = by_prefix.get(letter)
                if wids:
                    return [id_to_tok[w] for w in wids]
        by_prefix0 = tables[0].get(())
        if by_prefix0:
            wids = by_prefix0.get(letter)
            if wids:
                return [id_to_tok[w] for w in wids]
        return []
    return predict_topk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../weights/ngram_4_a.bin")
    ap.add_argument("--dev", default="../data/dev_set_final.csv")
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--max-prefix-len", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    n, counts, vocab, id_to_tok = load_model(args.model)
    print(f"loaded {args.model}: n={n}, vocab={len(vocab)}")
    print(f"building demo tables @ --min-count {args.min_count} "
          f"--max-prefix-len {args.max_prefix_len} --top-k {args.top_k}"
          f"{' --strict' if args.strict else ''}...")

    tables = {}
    for k in range(n):
        tables[k] = build_topk_table(counts[k], id_to_tok, args.min_count,
                                      args.max_prefix_len, args.top_k, strict=args.strict)
        print(f"  order {k}: {len(tables[k]):,} contexts")

    predict_topk = make_predictor(n, tables, vocab, id_to_tok)

    correct = defaultdict(int)
    total = defaultdict(int)
    with open(args.dev, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            answer, letter, context = row["answer"], row["first letter"], row["context"]
            cat = categorize(answer)
            total[cat] += 1
            if cat == "symbol":
                ok = (letter == answer)  # same trivial rule infer_ngram.py uses, no ngram involved
            else:
                preds = predict_topk(context.split(), letter)
                ok = answer in preds
            correct[cat] += ok

    print(f"\ntop-{args.top_k} accuracy (demo table, min-count={args.min_count}, "
          f"prefix-len={args.max_prefix_len}, strict={args.strict}):")
    overall_c, overall_t = sum(correct.values()), sum(total.values())
    print(f"overall: {overall_c}/{overall_t} = {overall_c/overall_t:.4f}")
    for cat in ("word", "symbol", "number"):
        c, t = correct[cat], total[cat]
        print(f"{cat}: {c}/{t} = {c/t:.4f}" if t else f"{cat}: 0/0")


if __name__ == "__main__":
    main()
