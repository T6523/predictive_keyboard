"""Split dev_set.csv 80/20 into a tuning set and an untouched holdout.

Does not modify dev_set.csv. Writes two new files:
    devv_eval.csv  (80%) -- hyperparameter tuning happens here
    devv_test.csv   (20%) -- final holdout, don't touch until the very end

Usage:
    python3 split_dev.py [--seed 42] [--frac 0.8]
"""
import argparse
import csv
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../data/dev_set.csv")
    ap.add_argument("--train-out", default="../data/devv_eval.csv")
    ap.add_argument("--test-out", default="../data/devv_test.csv")
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.src, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    rng = random.Random(args.seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    split = int(len(idx) * args.frac)
    train_idx, test_idx = set(idx[:split]), set(idx[split:])

    def write(path, keep):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i, row in enumerate(rows):
                if i in keep:
                    w.writerow(row)

    write(args.train_out, train_idx)
    write(args.test_out, test_idx)
    print(f"{len(rows)} rows -> {args.train_out}: {len(train_idx)}, {args.test_out}: {len(test_idx)}")


if __name__ == "__main__":
    main()
