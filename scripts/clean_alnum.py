"""Strip non-alnum tokens out of context text, so the n-gram only ever sees alphabet-or-number
tokens. Symbols and [UNK] first letters are answered deterministically instead (symbol_predict.py)
-- the n-gram never needs to model them, but every row is kept so scoring still counts them.

  - train.src.tok: the n-gram's training corpus. Lowercase, drop non-alnum tokens per line.
  - devv_eval.csv / devv_test.csv / test_set_no_answer.csv: eval/tune sets. No rows are ever
    dropped. Only 'context' tokens get the alnum filter (lowercased); 'first letter' and
    'answer' (where present) are left exactly as in the original, since scoring needs them
    unmodified -- symbol_predict.py answers the symbol/[UNK] rows, the n-gram answers the rest.

Originals in data/ are never touched. Output -> clean/.

Usage:
    python3 clean_alnum.py [--limit N]
"""
import argparse
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
CLEAN = Path(__file__).resolve().parent.parent / "clean"


def keep_tok(tok):
    return tok.isalnum()


def clean_corpus(src, dst, limit):
    n_lines = 0
    n_tokens_in = 0
    n_tokens_out = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            toks = line.lower().split()
            n_tokens_in += len(toks)
            kept = [t for t in toks if keep_tok(t)]
            n_tokens_out += len(kept)
            fout.write(" ".join(kept) + "\n")
            n_lines += 1
            if limit and n_lines >= limit:
                break
    return n_lines, n_tokens_in, n_tokens_out


def clean_context_only(src, dst, limit):
    """Filter context tokens to alnum; keep every row, leave answer/first letter untouched."""
    with open(src, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        fields = reader.fieldnames
        rows = []
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            toks = row["context"].lower().split()
            row["context"] = " ".join(t for t in toks if keep_tok(t))
            rows.append(row)
    with open(dst, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    CLEAN.mkdir(exist_ok=True)

    src = DATA / "train.src.tok"
    dst = CLEAN / "train.src.tok"
    n_lines, n_in, n_out = clean_corpus(src, dst, args.limit)
    print(f"train.src.tok: {n_lines} lines, tokens {n_in} -> {n_out} "
          f"(dropped {n_in - n_out}, {100*(n_in-n_out)/n_in:.1f}%)")
    print(f"  size: {src.stat().st_size/1e6:.1f}MB -> {dst.stat().st_size/1e6:.1f}MB")

    for name in ["devv_eval.csv", "devv_test.csv", "test_set_no_answer.csv"]:
        src = DATA / name
        dst = CLEAN / name
        n = clean_context_only(src, dst, args.limit)
        print(f"{name}: {n} rows (all kept) -> clean/{name}")
        print(f"  size: {src.stat().st_size/1e6:.1f}MB -> {dst.stat().st_size/1e6:.1f}MB")


if __name__ == "__main__":
    main()
