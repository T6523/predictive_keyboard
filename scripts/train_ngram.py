"""Train an n-gram language model on the cleaned train corpus (clean_alnum.py output) and save
counts as a .bin (pickle).

Usage:
    python3 train_ngram.py --n 3 --out ngram_3.bin
    python3 train_ngram.py --n 5 --out ngram_5.bin --min-count 2

Trains on data/train_final.src.tok (extracted from train_final.zip).

Model = plain count tables for order 1..n (so lower orders are free for backoff/interpolation
at inference time -- no smoothing baked in here, that's a tuning-stage decision).

Memory: tokens are encoded to int ids (vocab) instead of stored as repeated Python strings --
tuple-of-int context keys and int word keys are far cheaper than tuple-of-str, and dedupe
automatically instead of relying on string interning. Count tables are plain dicts (no
Counter/defaultdict object overhead). --min-count drops rare (context, word) pairs for
higher orders, applied once at the end by default (biggest further RAM lever, off by
default -- long-tail n-gram counts are mostly singletons, so this trims a lot but can lose
rare-but-correct hits).

--prune-every-lines makes that pruning happen periodically *during* the counting pass
instead of only once at the end -- needed for corpora too big to hold the full unpruned
count tables in RAM (e.g. gigaword: peak memory is hit while still counting, min-count
alone doesn't help since it's a one-shot filter applied after the peak). Trade-off: this
is approximate. A (context, word) pair that's below min_count at a prune checkpoint gets
evicted outright (no "keep if it would empty the context" fallback, unlike the final
pass) -- if it later recurs enough to pass the threshold, counting restarts from 0 for it,
so true counts can be undercounted for entries whose occurrences are spread far apart in
the stream. Fine for what pruning is for anyway (dropping rare stuff), not fine if you
need exact counts.
"""
import argparse
import pickle
import time

BOS, EOS = "<s>", "</s>"


def _compact(counts, n, min_count, hard):
    """Drop (ctx, word) entries below min_count for orders 1..n-1 (order 0 = unigrams,
    always kept intact). hard=True (streaming checkpoint): empty contexts are deleted
    outright, no fallback. hard=False (final pass): a context that would end up empty
    keeps its full unfiltered entry set instead, so backoff never hits a dead end."""
    for k in range(1, n):
        d_ctx = counts[k]
        for ctx in list(d_ctx.keys()):
            d = d_ctx[ctx]
            kept = {w: c for w, c in d.items() if c >= min_count}
            if kept:
                d_ctx[ctx] = kept
            elif hard:
                del d_ctx[ctx]
            else:
                d_ctx[ctx] = d  # keep everything rather than leave the context empty


def train(path, n, min_count=1, prune_every_lines=None):
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

            if (prune_every_lines and min_count > 1
                    and n_lines % prune_every_lines == 0):
                _compact(counts, n, min_count, hard=True)

    if min_count > 1:
        _compact(counts, n, min_count, hard=False)

    id_to_tok = [None] * len(vocab)
    for tok, i in vocab.items():
        id_to_tok[i] = tok

    return counts, vocab, id_to_tok, bos_id, eos_id, n_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/train_final.src.tok")
    ap.add_argument("--n", type=int, default=3, choices=range(3, 6), help="n-gram order (3-5)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-count", type=int, default=1,
                     help="drop (context,word) pairs with count below this, orders 2+ (RAM saver)")
    ap.add_argument("--prune-every-lines", type=int, default=None,
                     help="also compact counts every N lines during the pass, not just at "
                          "the end (needed for corpora too big to hold unpruned in RAM; "
                          "approximate, see module docstring)")
    args = ap.parse_args()
    out = args.out or f"../weights/ngram_{args.n}.bin"

    t0 = time.time()
    counts, vocab, id_to_tok, bos_id, eos_id, n_lines = train(
        args.train, args.n, args.min_count, args.prune_every_lines)

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
