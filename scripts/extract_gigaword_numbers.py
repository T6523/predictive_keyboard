"""Number-only slice of gigaword.tar.gz, anonymized to match train_final.src.tok's
digit scheme -- for training a separate number-length n-gram (status.md "To try":
Gigaword too slow to train in full, but a digit-only slice is tiny and fast).

Reuses tokenize_gigaword.py's streaming/paragraph/tokenize machinery. Two differences
from that script's full-corpus run:
  1. filters out paragraphs with no digit token at all ("grep only samples with num" --
     no point feeding the number n-gram lines it'll never be asked about)
  2. anonymizes digit tokens to "1"*len(token) BEFORE the vocab lookup, instead of
     letting them fall through to [UNK] -- train_final.src.tok's own digit tokens are
     already anonymized this way (confirmed: only "1", "11", ..., up to 8 digits show
     up, see report.md), so a literal Gigaword "37" needs the same transform to land in
     vocab at all, not a vocab-membership check against literal "37" (which was never
     going to be in train's vocab -- train has no real digit values left in it).

Usage:
    python3 extract_gigaword_numbers.py                # full run
    python3 extract_gigaword_numbers.py --limit 5       # quick test, first 5 tar members
"""
import argparse
import re
from pathlib import Path

from tokenize_gigaword import DATA, TOKEN_RE, load_train_vocab, paragraphs

DIGIT_RE = re.compile(r"^[0-9]+$")


def anonymize(tok):
    return "1" * len(tok) if DIGIT_RE.fullmatch(tok) else tok


def tokenize_line(line, vocab):
    toks = [anonymize(t.lower()) for t in TOKEN_RE.findall(line)]
    return toks


def demo():
    """Self-check: digit anonymization + has-a-number filter, no corpus needed."""
    assert anonymize("37") == "11"
    assert anonymize("1994") == "1111"
    assert anonymize("dog") == "dog"
    vocab = {"the", "dog", "1", "11", "."}
    toks = tokenize_line("The dog was 37 . xyzzy 9 years old", vocab)
    assert toks == ["the", "dog", "was", "11", ".", "xyzzy", "1", "years", "old"]
    assert any(DIGIT_RE.fullmatch(t) for t in toks)
    print("demo ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DATA / "gigaword.tar.gz"))
    ap.add_argument("--train-vocab", default=str(DATA / "train_final.src.tok"))
    ap.add_argument("--out", default=str(DATA / "gigaword_numbers.tok"))
    ap.add_argument("--limit", type=int, default=None, help="only first N tar members (test run)")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    import tarfile

    vocab = load_train_vocab(args.train_vocab)
    print(f"train vocab: {len(vocab)} tokens")

    n_files = n_lines_seen = n_lines_kept = n_tokens = n_unk = 0
    with tarfile.open(args.src, mode="r|gz") as tf, open(args.out, "w", encoding="utf-8") as fout:
        for member in tf:
            if not member.isfile():
                continue
            text = tf.extractfile(member).read().decode("utf-8", errors="replace")
            for p in paragraphs(text):
                n_lines_seen += 1
                toks = tokenize_line(p, vocab)
                if not toks or not any(DIGIT_RE.fullmatch(t) for t in toks):
                    continue  # no number in this paragraph -- skip, not useful for the number model
                out_toks = [t if (t in vocab or DIGIT_RE.fullmatch(t)) else "[UNK]" for t in toks]
                n_tokens += len(out_toks)
                n_unk += sum(1 for t in out_toks if t == "[UNK]")
                fout.write(" ".join(out_toks) + "\n")
                n_lines_kept += 1
            n_files += 1
            if args.limit and n_files >= args.limit:
                break
            if n_files % 20 == 0:
                print(f"[{n_files} files] {n_lines_kept}/{n_lines_seen} paragraphs kept", flush=True)

    print(f"\nfiles: {n_files}")
    print(f"paragraphs: {n_lines_kept}/{n_lines_seen} kept -> {args.out}")
    print(f"tokens: {n_tokens}, [UNK]: {n_unk} ({100 * n_unk / max(n_tokens,1):.2f}%)")


if __name__ == "__main__":
    main()
