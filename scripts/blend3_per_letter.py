"""Per-letter (lam_qwen, lam_mistral) instead of one global pair -- status.md "To try"
item 1. Hypothesis: high-branching first letters (r/s/c -- many common continuations)
vs peaked ones (t/o -- context pins the answer harder) may want different blend
weights. Free to try: blend3.py already precomputes per-row (qwen, mistral, kn5) score
triples, so this is just re-grouping the same argmax over letter subsets, no new
GPU/model calls.

Nested CV guard (per outer fold, before trusting per-letter on the held-out test fold):
  - split the outer fold's training rows into INNER_FOLDS
  - for each inner split: fit global lambda on inner-train, score inner-test (honest,
    out-of-sample) -- this is the baseline
  - same inner split: fit per-letter lambda on inner-train (backoff to that inner
    split's global lambda for letters below MIN_LETTER_SAMPLES), score inner-test
  - pool inner-test accuracy across all inner splits for both schemes -> "spread" =
    per-letter accuracy - global accuracy, an honest (out-of-sample) estimate
  - if spread < GUARD_PT, abandon per-letter for this outer fold -- use global lambda
    (refit on the FULL outer training set) for the outer test fold instead
  - if spread >= GUARD_PT, refit per-letter lambda on the full outer training set and
    use that for the outer test fold

Usage:
    python3 blend3_per_letter.py --qwen ../weights/qwen_tf_scores_k10.csv \
        --mistral ../weights/mistral_tf_scores_k10.csv
"""
import argparse
from collections import defaultdict

from blend3 import best_lambda, build_grid, eval_word, pick_best, precompute_rows
from blend_ngram import KN5Scorer, stratified_kfold
from eval_number_kn5 import pick_length

MIN_LETTER_SAMPLES = 50  # below this, backoff to the (inner- or outer-)global lambda
INNER_FOLDS = 4
GUARD_PT = 0.003  # +0.3pt -- per-letter must beat global by at least this, honestly, to keep


def fit_letter_lambdas(rows, grid, fallback_lam):
    """Group word rows by first_letter, grid-search each group with >=MIN_LETTER_SAMPLES
    rows, backoff the rest to fallback_lam. Returns {letter: (lam_q, lam_m)}."""
    by_letter = defaultdict(list)
    for row in rows:
        r = row[0]
        if r["category"] == "word":
            by_letter[r["first_letter"]].append(row)

    lams = {}
    for letter, group in by_letter.items():
        if len(group) >= MIN_LETTER_SAMPLES:
            lams[letter], _ = best_lambda(group, grid)
        else:
            lams[letter] = fallback_lam
    return lams


def eval_with_letter_lambdas(rows, letter_lams, fallback_lam):
    correct = total = 0
    for r, triples, _ in rows:
        if r["category"] != "word":
            continue
        lam = letter_lams.get(r["first_letter"], fallback_lam)
        pred = pick_best(triples, *lam)
        correct += (pred or "").strip().lower() == r["answer"].strip().lower()
        total += 1
    return correct, total


def demo():
    """Self-check: backoff kicks in below MIN_LETTER_SAMPLES, honest eval wiring."""
    fake_grid = build_grid(3)
    rows = []
    for i in range(60):  # >= MIN_LETTER_SAMPLES -- gets its own fit
        rows.append(({"category": "word", "first_letter": "a", "answer": "x"},
                      {"x": (0.0, 0.0, 0.0)}, None))
    for i in range(5):  # < MIN_LETTER_SAMPLES -- backs off
        rows.append(({"category": "word", "first_letter": "z", "answer": "x"},
                      {"x": (0.0, 0.0, 0.0)}, None))
    lams = fit_letter_lambdas(rows, fake_grid, fallback_lam=(0.42, 0.13))
    assert "a" in lams and lams["a"] != (0.42, 0.13)
    assert lams["z"] == (0.42, 0.13)
    print("demo ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", default="../weights/qwen_tf_scores_k10.csv")
    ap.add_argument("--mistral", default="../weights/mistral_tf_scores_k10.csv")
    ap.add_argument("--ngram", default="../weights/ngram_4_a.bin")
    ap.add_argument("--kn-model", default="../weights/kn5.binary")
    ap.add_argument("--number-model", default="../weights/kn5_numbers.binary")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lam-steps", type=int, default=11)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    print("precomputing candidate score triples...", flush=True)
    rows, _, ngram_best_of = precompute_rows(args.qwen, args.mistral, args.ngram, args.kn_model)
    number_scorer = KN5Scorer(args.number_model) if args.number_model else None
    print(f"done precomputing ({len(rows)} rows)", flush=True)

    grid = build_grid(args.lam_steps)
    outer_folds = stratified_kfold(rows, args.folds, args.split_seed, category_of=lambda row: row[0]["category"])

    cat_correct, cat_total = {}, {}
    spreads, kept = [], []
    for i in range(args.folds):
        test_fold = outer_folds[i]
        train_rows = [row for j, f in enumerate(outer_folds) if j != i for row in f]

        # nested inner CV: honest (out-of-sample) estimate of per-letter vs global,
        # using only this outer fold's training data
        inner_folds = stratified_kfold(train_rows, INNER_FOLDS, args.split_seed + 1000 + i,
                                        category_of=lambda row: row[0].get("first_letter", "?"))
        g_correct = g_total = pl_correct = pl_total = 0
        for j in range(INNER_FOLDS):
            inner_test = inner_folds[j]
            inner_train = [row for k, f in enumerate(inner_folds) if k != j for row in f]

            inner_global_lam, _ = best_lambda(inner_train, grid)
            c, t = eval_word(inner_test, *inner_global_lam)
            g_correct += c
            g_total += t

            inner_letter_lams = fit_letter_lambdas(inner_train, grid, inner_global_lam)
            c, t = eval_with_letter_lambdas(inner_test, inner_letter_lams, inner_global_lam)
            pl_correct += c
            pl_total += t

        global_acc = g_correct / g_total if g_total else 0
        per_letter_acc = pl_correct / pl_total if pl_total else 0
        spread = per_letter_acc - global_acc
        use_per_letter = spread >= GUARD_PT
        spreads.append(spread)
        kept.append(use_per_letter)

        # final fit on the FULL outer training set, applied to the held-out test fold
        global_lam, _ = best_lambda(train_rows, grid)
        letter_lams = fit_letter_lambdas(train_rows, grid, global_lam) if use_per_letter else {}

        for r, triples, ngram_best in test_fold:
            cat = r["category"]
            cat_total[cat] = cat_total.get(cat, 0) + 1
            if cat == "symbol":
                pred = r["first_letter"]
            elif cat == "number":
                pred = ("1" * pick_length(number_scorer, r["context"].split())
                        if number_scorer else ngram_best_of(r))
            else:
                lam = letter_lams.get(r["first_letter"], global_lam) if use_per_letter else global_lam
                pred = pick_best(triples, *lam)
            ok = (pred or "").strip().lower() == r["answer"].strip().lower()
            cat_correct[cat] = cat_correct.get(cat, 0) + ok

        print(f"fold {i+1}/{args.folds}: inner spread {spread*100:+.2f}pt "
              f"(global {global_acc*100:.2f}% vs per-letter {per_letter_acc*100:.2f}%) "
              f"-> {'KEPT per-letter' if use_per_letter else 'abandoned, using global'}", flush=True)

    print(f"\nper-fold inner spreads: {[f'{s*100:+.2f}pt' for s in spreads]}")
    print(f"per-fold kept per-letter: {kept}")
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
