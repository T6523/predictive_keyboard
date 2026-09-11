"""3-way blend (status.md advice item 4): Qwen teacher-forced log-prob + KN5 n-gram
log-prob + Mistral-7B teacher-forced log-prob, all on the SAME candidate set --
Mistral was scored directly against qwen_tf_scores.csv's own tf_preds column (see
score_candidates.py --lm-source tf), so this only adds one new term, no new
candidate-generation step.

score(w) = lam_qwen*qwen_tf(w) + lam_mistral*mistral_tf(w) + (1-lam_qwen-lam_mistral)*kn5(w)

Grid search over the (lam_qwen, lam_mistral) simplex, pooled stratified 5-fold CV
(same harness as blend_ngram.py). Per-row (candidate -> score triple) is precomputed
ONCE before the grid search -- KN5 scoring doesn't depend on lambda, so recomputing it
inside the grid loop (like the 1D blend_ngram.py does, tolerable at 21 steps) would be
disastrous at a 2D ~66-point grid x 5 folds.

Usage:
    python3 blend3.py --qwen ../weights/qwen_tf_scores.csv \
        --mistral ../weights/mistral_tf_scores.csv --ngram ../weights/ngram_4_a.bin \
        --kn-model ../weights/kn5.binary
"""
import argparse
import csv

from blend_ngram import FLOOR, KN5Scorer, _predict_word, parse_preds, stratified_kfold
from eval_number_kn5 import pick_length
from infer_ngram import load_model


def pick_best(triples, lam_q, lam_m):
    """triples: {word: (qwen_logprob, mistral_logprob, kn5_logprob)}. Argmax of the
    weighted sum."""
    lam_n = 1 - lam_q - lam_m
    best_word, best_score = None, float("-inf")
    for w, (q, m, k) in triples.items():
        s = lam_q * q + lam_m * m + lam_n * k
        if s > best_score:
            best_word, best_score = w, s
    return best_word


def build_grid(steps):
    """Triangular (lam_q, lam_m) grid with lam_q+lam_m<=1, `steps` points per axis."""
    return [(i / (steps - 1), j / (steps - 1)) for i in range(steps) for j in range(steps - i)]


def eval_word(rows, lam_q, lam_m):
    """rows: list of (r, triples, ngram_best). Accuracy of pick_best over the word rows
    in this set -- shared by blend3.py's own CV loop and blend3_per_letter.py's inner
    per-letter/global tuning (same shape, different row subsets)."""
    correct = total = 0
    for r, triples, _ in rows:
        if r["category"] != "word":
            continue
        pred = pick_best(triples, lam_q, lam_m)
        correct += (pred or "").strip().lower() == r["answer"].strip().lower()
        total += 1
    return correct, total


def best_lambda(rows, grid):
    """Grid-search argmax (lam_q, lam_m) by word accuracy on these rows."""
    best_lams, best_acc = (1.0, 0.0), -1
    for lam_q, lam_m in grid:
        c, t = eval_word(rows, lam_q, lam_m)
        acc = c / t if t else 0
        if acc > best_acc:
            best_lams, best_acc = (lam_q, lam_m), acc
    return best_lams, best_acc


def precompute_rows(qwen_path, mistral_path, ngram_path, kn_model_path):
    """Load qwen/mistral teacher-forced score CSVs + the n-gram model, build the
    per-row (candidate -> (qwen, mistral, kn5) log-prob triple) dict for every word
    row -- the expensive KN5-scoring step, done ONCE regardless of how many grid
    points/CV schemes get evaluated afterward. Returns (rows, ngram_model_tuple,
    number_scorer_ready_bits) -- callers needing number/ngram routing reuse the
    returned n/counts/vocab/id_to_tok instead of reloading."""
    n, counts, vocab, id_to_tok = load_model(ngram_path)
    kn_scorer = KN5Scorer(kn_model_path)

    qwen_rows = list(csv.DictReader(open(qwen_path, encoding="utf-8")))
    mistral_rows = list(csv.DictReader(open(mistral_path, encoding="utf-8")))
    assert len(qwen_rows) == len(mistral_rows), (len(qwen_rows), len(mistral_rows))

    def ngram_best_of(r):
        return _predict_word(r["context"].split(), r["first_letter"], n, counts, vocab, id_to_tok)

    rows = []
    for qr, mr in zip(qwen_rows, mistral_rows):
        assert qr["context"] == mr["context"] and qr["answer"] == mr["answer"], "row order mismatch between qwen/mistral CSVs"
        r = dict(qr)
        if qr["category"] != "word":
            rows.append((r, None, None))
            continue
        qwen_scores = dict(parse_preds(qr["tf_preds"]))
        mistral_scores = dict(parse_preds(mr["tf_preds"]))
        ngram_best = ngram_best_of(r)
        candidates = set(qwen_scores) | set(mistral_scores) | ({ngram_best} if ngram_best else set())
        ctx_tokens = r["context"].split()
        triples = {w: (qwen_scores.get(w, FLOOR), mistral_scores.get(w, FLOOR), kn_scorer.score(ctx_tokens, w))
                   for w in candidates}
        rows.append((r, triples, ngram_best))
    return rows, (n, counts, vocab, id_to_tok), ngram_best_of


def demo():
    """Self-check: pick_best argmax arithmetic, no model/kenlm needed."""
    triples = {"a": (-2.0, -1.0, -1.0), "b": (-1.0, -2.0, -5.0)}
    # lam_q=0.6, lam_m=0.4, lam_n=0 -> a: 0.6*-2+0.4*-1=-1.6  b: 0.6*-1+0.4*-2=-1.4 -> b wins
    assert pick_best(triples, 0.6, 0.4) == "b"
    # lam_q=0, lam_m=0, lam_n=1 (pure kn5) -> a:-1.0  b:-5.0 -> a wins
    assert pick_best(triples, 0.0, 0.0) == "a"
    assert build_grid(3) == [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5), (1.0, 0.0)]
    print("demo ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", default="../weights/qwen_tf_scores.csv")
    ap.add_argument("--mistral", default="../weights/mistral_tf_scores.csv")
    ap.add_argument("--ngram", default="../weights/ngram_4_a.bin")
    ap.add_argument("--kn-model", default="../weights/kn5.binary")
    ap.add_argument("--number-model", default="../weights/kn5_numbers.binary",
                     help="Gigaword+train number-only KN5 (74.59% vs the general n-gram's "
                          "71.97% on the same 2023-row set) -- pass '' to fall back to the "
                          "general n-gram for number routing")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lam-steps", type=int, default=11, help="grid points per axis, 0..1")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    print(f"precomputing candidate score triples...", flush=True)
    rows, (n, counts, vocab, id_to_tok), ngram_best_of = precompute_rows(
        args.qwen, args.mistral, args.ngram, args.kn_model)
    number_scorer = KN5Scorer(args.number_model) if args.number_model else None
    print(f"done precomputing ({len(rows)} rows)", flush=True)

    grid = build_grid(args.lam_steps)

    folds = stratified_kfold(rows, args.folds, args.split_seed, category_of=lambda row: row[0]["category"])
    print(f"\n{args.folds}-fold CV, {len(rows)} rows, {len(grid)} (lam_qwen, lam_mistral) grid points")

    cat_correct, cat_total = {}, {}
    chosen = []
    for i in range(args.folds):
        test_fold = folds[i]
        train_rows = [row for j, f in enumerate(folds) if j != i for row in f]

        best_lams, best_acc = best_lambda(train_rows, grid)
        chosen.append(best_lams)

        for r, triples, ngram_best in test_fold:
            cat = r["category"]
            cat_total[cat] = cat_total.get(cat, 0) + 1
            if cat == "symbol":
                pred = r["first_letter"]
            elif cat == "number":
                pred = ("1" * pick_length(number_scorer, r["context"].split())
                        if number_scorer else ngram_best_of(r))
            else:
                pred = pick_best(triples, *best_lams)
            ok = (pred or "").strip().lower() == r["answer"].strip().lower()
            cat_correct[cat] = cat_correct.get(cat, 0) + ok
        print(f"fold {i+1}/{args.folds}: lam_qwen={best_lams[0]:.2f} lam_mistral={best_lams[1]:.2f} "
              f"(train word acc {best_acc*100:.2f}%)", flush=True)

    print(f"\nper-fold chosen (lam_qwen, lam_mistral): {[(f'{q:.2f}', f'{m:.2f}') for q, m in chosen]}")
    print(f"\n--- pooled CV ({sum(cat_total.values())} rows) ---")
    tot_c = tot_n = 0
    for cat in ("word", "symbol", "number"):
        c, t = cat_correct.get(cat, 0), cat_total.get(cat, 0)
        tot_c += c
        tot_n += t
        if t:
            print(f"{cat:8s} {c:5d}/{t:5d}  {c/t*100:.2f}%")
    print(f"{'overall':8s} {tot_c:5d}/{tot_n:5d}  {tot_c/tot_n*100:.2f}%")


if __name__ == "__main__":
    main()
