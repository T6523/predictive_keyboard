"""Train an n-gram language model on the cleaned train corpus (clean_alnum.py output) and save
counts as a .bin (pickle).

Usage:
    python3 train_ngram.py --n 3 --out ngram_3.bin
    python3 train_ngram.py --n 5 --out ngram_5.bin --min-count 2

Trains on clean/train.src.tok (alnum-only tokens -- symbols/[UNK] are handled deterministically
by symbol_predict.py instead, see eval.py) not the raw data/train.src.tok.

Model = plain count tables for order 1..n (so lower orders are free for backoff/interpolation
at inference time -- no smoothing baked in here, that's a tuning-stage decision).

Memory: tokens are encoded to int ids (vocab) instead of stored as repeated Python strings --
tuple-of-int context keys and int word keys are far cheaper than tuple-of-str, and dedupe
automatically instead of relying on string interning. Count tables are plain dicts (no
Counter/defaultdict object overhead). --min-count drops rare (context, word) pairs for
higher orders as they're built (biggest further RAM lever, off by default -- long-tail
n-gram counts are mostly singletons, so this trims a lot but can lose rare-but-correct hits).
"""
import argparse
import pickle
import time

BOS, EOS = "<s>", "</s>"


def train(path, n, min_count=1):
    vocab = {}  # token -> id, assigned on first sight

    def tid(tok):
        i = vocab.get(tok)
        if i is None:
            i = vocab[tok] = len(vocab)
        return i

    bos_id, eos_id = tid(BOS), tid(EOS)

    # counts[k]: {ctx tuple(int, len k) -> {word_id: count}}, k = 0..n-1 (k=0 -> ctx = ())
    counts = [dict() for _ in range(n)]
    n_lines = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            ids = [bos_id] * (n - 1) + [tid(t) for t in line.split()] + [eos_id]
            n_lines += 1
            for i in range(n - 1, len(ids)):
                word = ids[i]
                for k in range(n):
                    ctx = tuple(ids[i - k:i])
                    d = counts[k].get(ctx)
                    if d is None:
                        counts[k][ctx] = d = {}
                    d[word] = d.get(word, 0) + 1

    if min_count > 1:
        for k in range(1, n):  # keep unigrams (k=0) intact -- always needed as final backoff
            for ctx, d in counts[k].items():
                counts[k][ctx] = {w: c for w, c in d.items() if c >= min_count} or d

    id_to_tok = [None] * len(vocab)
    for tok, i in vocab.items():
        id_to_tok[i] = tok

    return counts, vocab, id_to_tok, bos_id, eos_id, n_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../clean/train.src.tok")
    ap.add_argument("--n", type=int, default=3, choices=range(3, 6), help="n-gram order (3-5)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-count", type=int, default=1,
                     help="drop (context,word) pairs with count below this, orders 2+ (RAM saver)")
    args = ap.parse_args()
    out = args.out or f"../weights/ngram_{args.n}.bin"

    t0 = time.time()
    counts, vocab, id_to_tok, bos_id, eos_id, n_lines = train(args.train, args.n, args.min_count)

    model = {
        "n": args.n,
        "counts": counts,
        "vocab": vocab,
        "id_to_tok": id_to_tok,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "n_lines": n_lines,
    }
    with open(out, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"trained {args.n}-gram on {n_lines} lines in {time.time()-t0:.1f}s")
    print(f"saved -> {out}")
    print(f"  vocab: {len(vocab)} tokens")
    for k in range(args.n):
        print(f"  order {k+1}: {len(counts[k])} distinct contexts")


if __name__ == "__main__":
    main()
