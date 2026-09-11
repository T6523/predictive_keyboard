"""Build a (context, digit-length) dataset from train_final.src.tok for the number-length
classifier. Every number token in train is a run of '1's whose length encodes the original
digit count (see eda/vocab_coverage_eda.ipynb) -- label = len(token), context = every token
to its left on the same line (full sentence prefix, not just the last few tokens -- the
whole point of bag-of-words over an n-gram is using that).

Two sampling modes (both single-pass reservoir sampling, no need to know counts upfront):
  --mode balanced (default): reservoir per length, capped at --cap each, so rare lengths
    (5-8, ~0.02% of occurrences) aren't drowned out. Risk: the trained classifier no longer
    sees the true class frequencies (length 2 is ~39% of real occurrences, length 4 ~9%),
    so it can end up *worse* than an n-gram's frequency backoff on the real (imbalanced)
    distribution -- see dev_number_predictions_balanced.csv's result.
  --mode natural: one global reservoir capped at --cap total, sampled uniformly over all
    occurrences -- preserves the true class proportions, at the cost of the classifier
    seeing very few rare-length (5-8) examples (roughly proportional to how rare they
    actually are).

Usage:
    python3 extract_train_numbers.py --mode natural --out train_number_contexts_natural.csv
"""
import argparse
import csv
import random
import re
from pathlib import Path

NUM_RE = re.compile(r"^1+$")
DATA = Path(__file__).resolve().parent.parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(DATA / "train_final.src.tok"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "train_number_contexts.csv"))
    ap.add_argument("--mode", choices=["balanced", "natural"], default="balanced")
    ap.add_argument("--cap", type=int, default=150_000,
                     help="balanced: max contexts kept per length. natural: max total contexts kept.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen = {}  # length -> total occurrences seen so far (for reservoir sampling)

    if args.mode == "balanced":
        reservoir = {}  # length -> list[context str]
    else:
        global_reservoir = []  # list[(length, context str)]
        n_seen_total = 0

    with open(args.train, encoding="utf-8") as f:
        for line in f:
            toks = line.rstrip("\n").split()
            for i, t in enumerate(toks):
                if not NUM_RE.match(t) or i == 0:
                    continue  # skip sentence-initial numbers -- no context to learn from
                length = len(t)
                seen[length] = seen.get(length, 0) + 1
                ctx = " ".join(toks[:i])
                if args.mode == "balanced":
                    n = seen[length]
                    bucket = reservoir.setdefault(length, [])
                    if len(bucket) < args.cap:
                        bucket.append(ctx)
                    else:
                        j = rng.randint(0, n - 1)
                        if j < args.cap:
                            bucket[j] = ctx
                else:
                    n_seen_total += 1
                    if len(global_reservoir) < args.cap:
                        global_reservoir.append((length, ctx))
                    else:
                        j = rng.randint(0, n_seen_total - 1)
                        if j < args.cap:
                            global_reservoir[j] = (length, ctx)

    if args.mode == "natural":
        reservoir = {}
        for length, ctx in global_reservoir:
            reservoir.setdefault(length, []).append(ctx)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["context", "length"])
        for length in sorted(reservoir):
            for ctx in reservoir[length]:
                w.writerow([ctx, length])

    print("occurrences seen per length:", dict(sorted(seen.items())))
    print("kept per length:", {k: len(v) for k, v in sorted(reservoir.items())})
    print("total kept:", sum(len(v) for v in reservoir.values()), "->", args.out)


if __name__ == "__main__":
    main()
