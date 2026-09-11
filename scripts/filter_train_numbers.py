"""Number-only slice of train_final.src.tok, same filter as extract_gigaword_numbers.py
applies to Gigaword -- keep only lines with >=1 digit token. No anonymization or vocab
masking needed here (train_final.src.tok is already both), just the filter, so this run
in ~seconds vs Gigaword's ~28min tar-stream.

Usage:
    python3 filter_train_numbers.py
"""
import argparse
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DIGIT_RE = re.compile(r"^[0-9]+$")


def demo():
    assert any(DIGIT_RE.fullmatch(t) for t in "the dog was 11 years old".split())
    assert not any(DIGIT_RE.fullmatch(t) for t in "the dog was old".split())
    print("demo ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DATA / "train_final.src.tok"))
    ap.add_argument("--out", default=str(DATA / "train_numbers.tok"))
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    n_seen = n_kept = 0
    with open(args.src, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            n_seen += 1
            if any(DIGIT_RE.fullmatch(t) for t in line.split()):
                fout.write(line)
                n_kept += 1

    print(f"lines: {n_kept}/{n_seen} kept -> {args.out}")


if __name__ == "__main__":
    main()
