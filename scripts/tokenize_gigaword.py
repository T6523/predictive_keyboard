"""Tokenize gigaword.tar.gz onto train.src.tok's exact vocab.

Streams straight from the tar.gz (never extracted to disk). Each blank-line-delimited
paragraph becomes one output line: lowercased, punctuation isolated the same way
train.src.tok does it ("o.j." -> "o . j .", "don't" -> "don ' t", "well-known" ->
"well - known", "-LRB-" -> "- lrb -"), any resulting token not in train.src.tok's
vocab replaced with [UNK]. No stripping of symbols/punctuation -- unlike
clean_alnum.py, everything survives as a token, just possibly [UNK].

Output goes to data/ (not clean/) since it's a derived-from-raw artifact, not a
cleaned version of an existing clean/ file.

Usage:
    python3 tokenize_gigaword.py                # full run (~10min, one streaming pass)
    python3 tokenize_gigaword.py --limit 2       # quick test on first 2 monthly files
"""
import argparse
import re
import tarfile
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# train.src.tok isolates every punctuation character as its own token (e.g. "o.j." ->
# "o . j .", "don't" -> "don ' t", "well-known" -> "well - known", "-LRB-" -> "- lrb -").
# Reproduce that exact split so gigaword tokens actually land in the vocab instead of
# spuriously mismatching on attached punctuation.
TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")


def load_train_vocab(path):
    vocab = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            vocab.update(line.split())
    return vocab


def paragraphs(text):
    """Blank-line-delimited paragraphs, wrapped lines rejoined -- same unit used in
    eda/gigaword_eda.ipynb."""
    para, out = [], []
    for line in text.split("\n"):
        line = line.strip()
        if line == "":
            if para:
                out.append(" ".join(para))
                para = []
        else:
            para.append(line)
    if para:
        out.append(" ".join(para))
    return out


def tokenize_line(line, vocab):
    toks = [t.lower() for t in TOKEN_RE.findall(line)]
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DATA / "gigaword.tar.gz"))
    ap.add_argument("--train-vocab", default=str(DATA / "train.src.tok"))
    ap.add_argument("--out", default=str(DATA / "gigaword.tok"))
    ap.add_argument("--limit", type=int, default=None,
                     help="only process the first N tar members (quick test run)")
    args = ap.parse_args()

    vocab = load_train_vocab(args.train_vocab)
    print(f"train vocab: {len(vocab)} tokens")

    n_files = 0
    n_lines = 0
    n_tokens = 0
    n_unk = 0
    with tarfile.open(args.src, mode="r|gz") as tf, open(args.out, "w", encoding="utf-8") as fout:
        for member in tf:
            if not member.isfile():
                continue
            text = tf.extractfile(member).read().decode("utf-8", errors="replace")
            for p in paragraphs(text):
                toks = tokenize_line(p, vocab)
                if not toks:
                    continue
                out_toks = [t if t in vocab else "[UNK]" for t in toks]
                n_tokens += len(out_toks)
                n_unk += sum(1 for t in out_toks if t == "[UNK]")
                fout.write(" ".join(out_toks) + "\n")
                n_lines += 1
            n_files += 1
            if args.limit and n_files >= args.limit:
                break

    print(f"files: {n_files}")
    print(f"lines written: {n_lines} -> {args.out}")
    print(f"tokens: {n_tokens}, [UNK]: {n_unk} ({100 * n_unk / n_tokens:.2f}%)")


if __name__ == "__main__":
    main()
