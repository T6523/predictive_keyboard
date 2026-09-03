"""Train an n-gram language model on the cleaned train corpus (clean_alnum.py output) and save
counts as a .bin (pickle).

Usage:
    python3 train_ngram.py --n 3 --out ngram_3.bin
    python3 train_ngram.py --n 5 --out ngram_5.bin

Trains on clean/train.src.tok (alnum-only tokens -- symbols/[UNK] are handled deterministically
by symbol_predict.py instead, see eval.py) not the raw data/train.src.tok.

Model = plain count tables for order 1..n (so lower orders are free for backoff/interpolation
at inference time -- no smoothing baked in here, that's a tuning-stage decision).
"""
import argparse
import pickle
import time
from collections import defaultdict, Counter

BOS, EOS = "<s>", "</s>"


def train(path, n):
    # counts[k] = Counter mapping a k-token context tuple -> Counter of next-token counts
    # k ranges 0..n-1 (k=0 is unigram: context is the empty tuple)
    counts = {k: defaultdict(Counter) for k in range(n)}
    n_lines = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            toks = [BOS] * (n - 1) + line.split() + [EOS]
            n_lines += 1
            for i in range(n - 1, len(toks)):
                word = toks[i]
                for k in range(n):
                    ctx = tuple(toks[i - k:i])
                    counts[k][ctx][word] += 1
    return counts, n_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../clean/train.src.tok")
    ap.add_argument("--n", type=int, default=3, choices=range(3, 6), help="n-gram order (3-5)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"../weights/ngram_{args.n}.bin"

    t0 = time.time()
    counts, n_lines = train(args.train, args.n)
    counts = {k: dict(v) for k, v in counts.items()}  # drop defaultdict for portable pickle

    model = {
        "n": args.n,
        "counts": counts,
        "bos": BOS,
        "eos": EOS,
        "n_lines": n_lines,
    }
    with open(out, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"trained {args.n}-gram on {n_lines} lines in {time.time()-t0:.1f}s")
    print(f"saved -> {out}")
    for k in sorted(counts):
        print(f"  order {k+1}: {len(counts[k])} distinct contexts")


if __name__ == "__main__":
    main()
