"""Clean tokenization artifacts found during EDA, without touching the originals.

Fixes:
  - PTB bracket tokens split with stray spaces: "- lrb -" -> "-lrb-" (also rrb/lcb/rcb/lsb/rsb)
  - lowercases everything

Applies to train.src.tok (line corpus) and the devv_eval / devv_test / test_set_no_answer
csvs (their 'context' and, if present, 'answer' columns).

Usage (small batch first, per instructions -- don't full-run yet):
    python3 clean_pipeline.py --limit 2000
    python3 clean_pipeline.py            # full run, once the sample looks right
"""
import argparse
import csv
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
CLEAN = Path(__file__).resolve().parent.parent / "clean"

BRACKET_RE = re.compile(r"-\s+(lrb|rrb|lcb|rcb|lsb|rsb)\s+-")


def clean_text(s):
    s = s.lower()
    s = BRACKET_RE.sub(r"-\1-", s)
    return s


def clean_corpus(src, dst, limit):
    n = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(clean_text(line.rstrip("\n")) + "\n")
            n += 1
            if limit and n >= limit:
                break
    return n


def clean_csv(src, dst, limit):
    with open(src, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        fields = reader.fieldnames
        rows = []
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            if "context" in row:
                row["context"] = clean_text(row["context"])
            if "answer" in row:
                row["answer"] = clean_text(row["answer"])
            rows.append(row)
    with open(dst, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="rows/lines per file (small-batch test run)")
    args = ap.parse_args()

    CLEAN.mkdir(exist_ok=True)

    n = clean_corpus(DATA / "train.src.tok", CLEAN / "train.src.tok", args.limit)
    print(f"train.src.tok: {n} lines -> clean/train.src.tok")

    n = clean_csv(DATA / "devv_eval.csv", CLEAN / "devv_eval.csv", args.limit)
    print(f"devv_eval.csv: {n} rows -> clean/devv_eval.csv")

    n = clean_csv(DATA / "devv_test.csv", CLEAN / "devv_test.csv", args.limit)
    print(f"devv_test.csv: {n} rows -> clean/devv_test.csv")

    n = clean_csv(DATA / "test_set_no_answer.csv", CLEAN / "test_set_no_answer.csv", args.limit)
    print(f"test_set_no_answer.csv: {n} rows -> clean/test_set_no_answer.csv")


if __name__ == "__main__":
    main()
