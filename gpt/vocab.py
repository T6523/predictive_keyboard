"""Word-level vocab. build_notebook.py points this at whatever train.src.tok got uploaded
to the Kaggle dataset (glob on /kaggle/input/*/train.src.tok) -- in practice that's the
*raw* data/train.src.tok (99021 unique tokens, symbols kept), not the alnum-stripped
clean/train.src.tok the n-gram trains on. With MIN_COUNT=1 and BOS/EOS added below, the
checkpoint's actual vocab_size is 99023. Kept standalone rather than reusing
weights/ngram_N.counts.pkl so this folder doesn't need the (large) n-gram artifact just to
get a token->id map.

Also builds the letter-bucket mask used for masked training/eval: every word's first
character (a-z, 0-9) buckets it, since the task always gives the label's first letter --
training restricts the softmax to same-bucket words at every position, not just eval time.
"""
import string

BOS, EOS = "<s>", "</s>"
BUCKETS = list(string.ascii_lowercase) + list(string.digits)  # 36 buckets


def build_vocab(train_path, min_count=1):
    freq = {}
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            for tok in line.split():
                freq[tok] = freq.get(tok, 0) + 1

    vocab = {BOS: 0, EOS: 1}
    for tok, c in freq.items():
        if c >= min_count:
            vocab[tok] = len(vocab)
    id_to_tok = [None] * len(vocab)
    for tok, i in vocab.items():
        id_to_tok[i] = tok
    return vocab, id_to_tok


def bucket_of(tok):
    """Which letter-bucket a word belongs to, or None (BOS/EOS/anything not alnum-first)."""
    if not tok or not tok[0].isalnum():
        return None
    ch = tok[0].lower()
    return BUCKETS.index(ch) if ch in BUCKETS else None


def build_bucket_table(id_to_tok):
    """id -> bucket index (or -1 for BOS/EOS/no-bucket), for building the mask tensor."""
    table = []
    for tok in id_to_tok:
        b = bucket_of(tok)
        table.append(b if b is not None else -1)
    return table
