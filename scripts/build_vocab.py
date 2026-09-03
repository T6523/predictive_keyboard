"""Dump the unique word list from the training corpus, for eval_ngram.py to filter
next-word candidates by first letter (KenLM's Python binding doesn't expose vocab
enumeration, only word->id lookup). One file, shared by every ngram_N.bin trained on
the same corpus.

Usage:
    python3 build_vocab.py --train ../clean/train.src.tok --out ../weights/vocab.txt
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../clean/train.src.tok")
    ap.add_argument("--out", default="../weights/vocab.txt")
    args = ap.parse_args()

    vocab = set()
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            vocab.update(line.split())

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(vocab)) + "\n")

    print(f"{len(vocab)} unique words -> {args.out}")


if __name__ == "__main__":
    main()
