"""Build the actual files the GitHub Pages demo ships: one gzip'd trie per n-gram order plus
a gzip'd vocab (id -> token), using the exact config validated in export_demo_model.py
(--min-count 10 --strict --max-prefix-len 1 --top-k 3 -> ~10MB gzip total, 70% top-3 dev
accuracy, #1 prediction always identical to real predict_word).

Reuses build_streaming (the memory-safe path) and encode_trie (the format the JS reader in
docs/app.js parses byte-for-byte) directly -- no new logic here, just writing what was already
measured out to disk.

Usage:
    python3 build_release_assets.py --model ../weights/ngram_4_a.bin --out ../docs/data
"""
import argparse
import gzip
import pickle
from pathlib import Path

from export_demo_model import build_streaming, encode_trie


def load_model(path):
    with open(path, "rb") as f:
        m = pickle.load(f)
    return m["n"], m["counts"], m["id_to_tok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../weights/ngram_4_a.bin")
    ap.add_argument("--out", default="../docs/data")
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--max-prefix-len", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    n, counts, id_to_tok = load_model(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in range(n):
        _, trie = build_streaming(counts[k], id_to_tok, args.min_count, args.max_prefix_len,
                                   args.top_k, want_trie=True, strict=True)
        counts[k] = None
        trie_buf = encode_trie(trie)
        del trie
        path = out_dir / f"order{k}.trie.gz"
        path.write_bytes(gzip.compress(trie_buf, compresslevel=9))
        print(f"order {k}: {len(trie_buf)/1e6:.2f} MB raw -> {path} "
              f"({path.stat().st_size/1e6:.2f} MB gzip)")

    vocab_bytes = "\n".join(t if t else "" for t in id_to_tok).encode("utf-8")
    vocab_path = out_dir / "vocab.txt.gz"
    vocab_path.write_bytes(gzip.compress(vocab_bytes, compresslevel=9))
    print(f"vocab: {len(id_to_tok):,} tokens -> {vocab_path} "
          f"({vocab_path.stat().st_size/1e6:.2f} MB gzip)")


if __name__ == "__main__":
    main()
