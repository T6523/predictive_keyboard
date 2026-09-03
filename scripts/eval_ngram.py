"""Score an n-gram model on a dev-style csv (context, first letter, answer).

Stupid backoff: try the highest order context first, among next-token candidates *actually
observed* after that context, keep only ones starting with the required letter, take the
highest raw count. Falls back one order down on a miss, down to unigrams.

Model = weights/ngram_N.counts.pkl, from train_ngram.py -- int-encoded raw counts, not the
KenLM-trained weights/ngram_N.bin. Ranking needs exact counts (which word was seen most often
after this context), not a smoothed probability: KenLM's Kneser-Ney backoff reorders candidates
by adding a per-word backoff-weighted lower-order term, which can rank a common word above the
true highest-count continuation -- fine for perplexity, wrong for this exact-match metric. See
weights/ngram_N.bin (trained by lmplz, scripts/build_kenlm.sh) for the compact/mmap'd artifact;
it isn't used for prediction here.

Usage:
    python3 eval_ngram.py --model ../weights/ngram_3.counts.pkl --data ../clean/devv_eval.csv
    python3 eval_ngram.py --model ../weights/ngram_3.counts.pkl --data ../clean/devv_eval.csv --limit 2000
"""
import argparse
import csv
import pickle
import time


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(model, context_tokens, letter):
    n, counts, vocab, id_to_tok, bos_id = (
        model["n"], model["counts"], model["vocab"], model["id_to_tok"], model["bos_id"]
    )
    # -1 for unseen tokens: never matches any stored context, so it backs off/misses same as
    # a raw unseen string would.
    padded = [bos_id] * (n - 1) + [vocab.get(t, -1) for t in context_tokens]
    letter = letter.lower()  # vocab is lowercased at train time
    unigrams = counts[0].get(())  # word_id -> global count, tie-break for equal-count candidates
    for k in range(n - 1, -1, -1):  # k = context length used, from n-1 down to 0
        ctx = tuple(padded[len(padded) - k:]) if k else ()
        cands = counts[k].get(ctx)
        if not cands:
            continue
        best_word, best_c, best_u = None, -1, -1
        for word, c in cands.items():
            if not id_to_tok[word].startswith(letter):
                continue
            u = unigrams.get(word, 0) if unigrams else 0
            if c > best_c or (c == best_c and u > best_u):
                best_word, best_c, best_u = word, c, u
        if best_word is not None:
            return id_to_tok[best_word], k  # k = order that produced the hit
    return None, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../weights/ngram_3.counts.pkl")
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
