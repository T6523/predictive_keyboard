"""Score an n-gram model (from train_ngram.py) on a dev-style csv (context, first letter, answer).

Stupid backoff: try the highest order context first: among next-token candidates seen after
that context, keep only ones starting with the required letter, take the highest count. If no
candidate survives (context unseen, or nothing there starts with that letter) fall back one
order down, down to unigrams.

Usage:
    python3 eval_ngram.py --model ngram_3.bin --data devv_eval.csv
    python3 eval_ngram.py --model ngram_3.bin --data devv_eval.csv --limit 2000
"""
import argparse
import csv
import pickle
import time


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(model, context_tokens, letter):
    n, counts, bos = model["n"], model["counts"], model["bos"]
    padded = [bos] * (n - 1) + context_tokens
    for k in range(n - 1, -1, -1):  # k = context length used, from n-1 down to 0
        ctx = tuple(padded[len(padded) - k:]) if k else ()
        cands = counts[k].get(ctx)
        if not cands:
            continue
        best_tok, best_c = None, -1
        for tok, c in cands.items():
            if tok.startswith(letter) and c > best_c:
                best_tok, best_c = tok, c
        if best_tok is not None:
            return best_tok, k  # k = order that produced the hit
    return None, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../weights/ngram_3.bin")
    ap.add_argument("--data", default="../data/devv_eval.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model = load_model(args.model)
    print(f"loaded {model['n']}-gram model from {args.model}")

    with open(args.data, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    correct = 0
    no_pred = 0
    hits_by_order = {}
    t0 = time.time()
    for row in rows:
        toks = row["context"].split()
        letter = row["first letter"]
        answer = row["answer"]
        pred, order = predict(model, toks, letter)
        if pred is None:
            no_pred += 1
        else:
            hits_by_order[order] = hits_by_order.get(order, 0) + 1
            if pred == answer:
                correct += 1

    n_total = len(rows)
    print(f"rows: {n_total}")
    print(f"accuracy: {correct}/{n_total} = {correct / n_total:.4f}")
    print(f"no prediction (no candidate at any order): {no_pred}")
    print(f"predictions by backoff order used: {dict(sorted(hits_by_order.items(), reverse=True))}")
    print(f"eval time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
