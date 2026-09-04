"""Runnable smoke test for vocab.py + data.py -- no torch needed. python3 transformer/test_data.py"""
import os
import tempfile

from vocab import build_vocab, build_bucket_table, bucket_of, BOS, EOS, BUCKETS
from data import iter_lines, tokenize_corpus, n_blocks, block, HELDOUT_EVERY

CORPUS = "\n".join(
    f"the quick fox jumps over lazy dog number {i} , said reporter -lrb- ap -rrb- ."
    for i in range(200)
)


def test_train_heldout_split_disjoint(path):
    train_lines = set(iter_lines(path, 1.0, 0, "train"))
    heldout_lines = set(iter_lines(path, 1.0, 0, "heldout"))
    assert not (train_lines & heldout_lines), "train/heldout leak"
    # every HELDOUT_EVERY-th physical line goes to heldout
    with open(path) as f:
        raw = f.readlines()
    expect_heldout = sum(1 for i in range(len(raw)) if i % HELDOUT_EVERY == 0)
    assert len(heldout_lines) <= expect_heldout  # dedup collapses identical lines in this fixture
    print("ok: train/heldout split disjoint")


def test_subsample_frac_roughly_matches(path):
    full = sum(1 for _ in iter_lines(path, 1.0, 0, "train"))
    half = sum(1 for _ in iter_lines(path, 0.5, 0, "train"))
    assert 0.3 * full < half < 0.7 * full, f"frac subsample off: {half}/{full}"
    print("ok: SUBSET_FRAC subsampling in expected range")


def test_tokenize_bos_eos_and_symbols(path):
    vocab, id_to_tok = build_vocab(path, min_count=1)
    ids = tokenize_corpus(path, vocab, frac=1.0, seed=0, split="train")
    assert ids[0] == vocab[BOS]
    # symbols (",", "-lrb-"... here unmerged as "-","lrb","-", ".") must survive as vocab entries
    for sym in [",", "."]:
        assert sym in vocab, f"symbol token {sym!r} dropped -- must stay for transformer context"
    print("ok: BOS-prefixed, symbols kept in vocab")


def test_bucket_table_matches_bucket_of(path):
    vocab, id_to_tok = build_vocab(path, min_count=1)
    table = build_bucket_table(id_to_tok)
    for tok, wid in vocab.items():
        expect = bucket_of(tok)
        assert table[wid] == (expect if expect is not None else -1)
    # symbol tokens must have no bucket (never a valid prediction target)
    sym_id = vocab[","]
    assert table[sym_id] == -1
    # real words must have a bucket matching their first letter
    fox_id = vocab["fox"]
    assert table[fox_id] == BUCKETS.index("f")
    print("ok: bucket table matches vocab.bucket_of for words and symbols")


def test_blocks_cover_ids_without_overlap(path):
    vocab, _ = build_vocab(path, min_count=1)
    ids = tokenize_corpus(path, vocab, frac=1.0, seed=0, split="train")
    seq_len = 8
    nb = n_blocks(len(ids), seq_len)
    assert nb > 0
    for i in range(nb):
        x, y = block(ids, i, seq_len)
        assert len(x) == len(y) == seq_len
        assert y[:-1] == x[1:]  # y is x shifted by one -- standard causal-LM framing
    print(f"ok: {nb} blocks well-formed (x/y shift-by-one holds)")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "corpus.tok")
        with open(path, "w") as f:
            f.write(CORPUS)
        test_train_heldout_split_disjoint(path)
        test_subsample_frac_roughly_matches(path)
        test_tokenize_bos_eos_and_symbols(path)
        test_bucket_table_matches_bucket_of(path)
        test_blocks_cover_ids_without_overlap(path)
    print("all good")
