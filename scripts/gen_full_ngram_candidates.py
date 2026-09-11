"""N-gram top-10 candidates for the FULL dev set (all 94488 rows, not the 10%
stratified sample) -- CPU-only, no GPU needed. Feeds the Kaggle full-dev-set beam-k10
run: upload this CSV as a Kaggle input dataset so the notebook can union its own beam
candidates with these without needing ngram_4_a.bin (a local-only binary) on Kaggle.

Usage:
    python3 gen_full_ngram_candidates.py --out ../weights/dev_full_ngram_top10.csv
"""
import argparse
import csv
import time

from infer_ngram import categorize, load_model, topk_by_letter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="../data/dev_set_final.csv")
    ap.add_argument("--ngram", default="../weights/ngram_4_a.bin")
    ap.add_argument("--ngram-topk", type=int, default=10)
    ap.add_argument("--out", default="../weights/dev_full_ngram_top10.csv")
    args = ap.parse_args()

    n, counts, vocab, id_to_tok = load_model(args.ngram)

    with open(args.dev, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} dev rows total")

    t0 = time.time()
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["context", "first_letter", "answer", "category", "ngram_top10"])
        for i, r in enumerate(rows):
            letter = r["first letter"]
            cat = categorize(r["answer"])
            if cat == "word" or cat == "number":
                cands = topk_by_letter(r["context"].split(), letter, n, counts, vocab, id_to_tok, args.ngram_topk)
            else:
                cands = []
            w.writerow([r["context"], letter, r["answer"], cat, "; ".join(cands)])
            if (i + 1) % 20000 == 0:
                print(f"[{i+1}/{len(rows)}] {time.time()-t0:.1f}s", flush=True)

    print(f"done in {time.time()-t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
