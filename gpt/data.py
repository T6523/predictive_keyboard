"""Corpus loading for train.ipynb -- pure python/numpy (no torch), so it's testable without
a GPU env. See test_data.py for the runnable check.

Held-out split: every HELDOUT_EVERY-th line (by index) is reserved for eval-loss tracking
during training (needed to actually see a double-descent curve -- can't see it from final
numbers alone). Independent of SUBSET_FRAC, which then subsamples *within* whichever split so
a fast test run and a full run share the same code path -- just flip SUBSET_FRAC.
"""
import random

from vocab import BOS, EOS

HELDOUT_EVERY = 50  # ~2% of lines held out, never trained on


def iter_lines(path, frac, seed, split):
    """Yield raw lines for one split ('train' or 'heldout'), deterministically subsampled to
    `frac` of that split (frac=1.0 -> everything)."""
    is_train = split == "train"
    rng = random.Random(seed if is_train else seed + 1)
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if (i % HELDOUT_EVERY == 0) == is_train:
                continue  # wrong split for this call
            if frac < 1.0 and rng.random() >= frac:
                continue
            yield line


def tokenize_corpus(path, vocab, frac=1.0, seed=0, split="train"):
    """Whitespace/lowercase tokenize (source is already lowercase; .lower() is a cheap no-op
    safety net) -- no alnum filter, no bracket-merge. Symbols stay in as context tokens;
    vocab.bucket_of() already routes them to no-bucket so they're never a prediction target.
    OOV (below MIN_COUNT) -> BOS id, rare/ignorable at MIN_COUNT=1."""
    bos_id, eos_id = vocab[BOS], vocab[EOS]
    ids = []
    for line in iter_lines(path, frac, seed, split):
        ids.append(bos_id)
        for tok in line.lower().split():
            ids.append(vocab.get(tok, bos_id))
        ids.append(eos_id)
    return ids


def n_blocks(n_ids, seq_len):
    return (n_ids - 1) // seq_len


def block(ids, i, seq_len):
    """i-th fixed-size (x, y) block as plain lists -- notebook wraps ids in a torch tensor and
    slices the same way inside a Dataset.__getitem__."""
    s = i * seq_len
    chunk = ids[s : s + seq_len + 1]
    return chunk[:-1], chunk[1:]
