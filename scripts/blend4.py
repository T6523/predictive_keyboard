"""4-way blend (status.md "To try" -- vet a 4th teacher-forced scorer): Qwen tf +
Mistral tf + granite-3.1-2b tf + KN5, all on the SAME candidate set (granite scored
directly against qwen_tf_scores_k10.csv's tf_preds column via
score_candidates.py --lm-source tf, same trick as Mistral -- no new candidate
generation). Mirrors blend3.py exactly, one more term.

score(w) = lam_q*qwen_tf(w) + lam_m*mistral_tf(w) + lam_g*granite_tf(w)
           + (1-lam_q-lam_m-lam_g)*kn5(w)

3D simplex grid search (lam_q+lam_m+lam_g<=1), pooled stratified 5-fold CV. Keep only
if granite's lam_g > 0.1 AND CV word-accuracy gain over blend3.py's 3-way > +0.3pt --
otherwise granite isn't pulling its weight and adds a free parameter for nothing.

Usage:
    python3 blend4.py --qwen ../weights/qwen_tf_scores_k10.csv \
        --mistral ../weights/mistral_tf_scores_k10.csv --granite ../weights/granite_tf_scores_k10.csv
"""
import argparse
import csv

from blend_ngram import FLOOR, KN5Scorer, _predict_word, parse_preds, stratified_kfold
from eval_number_kn5 import pick_length
from infer_ngram import load_model


def pick_best(quads, lam_q, lam_m, lam_g):
    """quads: {word: (qwen_logprob, mistral_logprob, granite_logprob, kn5_logprob)}."""
    lam_n = 1 - lam_q - lam_m - lam_g
    best_word, best_score = None, float("-inf")
    for w, (q, m, g, k) in quads.items():
        s = lam_q * q + lam_m * m + lam_g * g + lam_n * k
        if s > best_score:
            best_word, best_score = w, s
    return best_word


def build_grid(steps):
    """Triangular (lam_q, lam_m, lam_g) simplex grid, lam_q+lam_m+lam_g<=1."""
    return [(i / (steps - 1), j / (steps - 1), l / (steps - 1))
            for i in range(steps) for j in range(steps - i) for l in range(steps - i - j)]


def eval_word(rows, lam_q, lam_m, lam_g):
    correct = total = 0
    for r, quads, _ in rows:
        if r["category"] != "word":
            continue
        pred = pick_best(quads, lam_q, lam_m, lam_g)
        correct += (pred or "").strip().lower() == r["answer"].strip().lower()
        total += 1
    return correct, total


def best_lambda(rows, grid):
    best_lams, best_acc = (1.0, 0.0, 0.0), -1
    for lam_q, lam_m, lam_g in grid:
        c, t = eval_word(rows, lam_q, lam_m, lam_g)
        acc = c / t if t else 0
        if acc > best_acc:
            best_lams, best_acc = (lam_q, lam_m, lam_g), acc
    return best_lams, best_acc


def demo():
    """Self-check: pick_best argmax arithmetic + grid shape, no model/kenlm needed."""
    quads = {"a": (-2.0, -1.0, -3.0, -1.0), "b": (-1.0, -2.0, -1.0, -5.0)}
    # lam_q=0.5,lam_m=0.3,lam_g=0.2,lam_n=0 -> a: .5*-2+.3*-1+.2*-3=-2.2  b: .5*-1+.3*-2+.2*-1=-1.3 -> b
    assert pick_best(quads, 0.5, 0.3, 0.2) == "b"
    # pure granite (lam_g=1) -> a:-3.0 b:-1.0 -> b
    assert pick_best(quads, 0.0, 0.0, 1.0) == "b"
    grid = build_grid(3)
    assert all(i + j + l <= 2 for i, j, l in [(round(a*2), round(b*2), round(c*2)) for a, b, c in grid])
    print("demo ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", default="../weights/qwen_tf_scores_k10.csv")
    ap.add_argument("--mistral", default="../weights/mistral_tf_scores_k10.csv")
    ap.add_argument("--granite", default="../weights/granite_tf_scores_k10.csv")
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

    n, counts, vocab, id_to_tok = load_model(args.ngram)
    kn_scorer = KN5Scorer(args.kn_model)
    number_scorer = KN5Scorer(args.number_model) if args.number_model else None

    qwen_rows = list(csv.DictReader(open(args.qwen, encoding="utf-8")))
    mistral_rows = list(csv.DictReader(open(args.mistral, encoding="utf-8")))
    granite_rows = list(csv.DictReader(open(args.granite, encoding="utf-8")))
    assert len(qwen_rows) == len(mistral_rows) == len(granite_rows)

    def ngram_best_of(r):
        return _predict_word(r["context"].split(), r["first_letter"], n, counts, vocab, id_to_tok)

    print("precomputing candidate score quadruples...", flush=True)
    rows = []
    for qr, mr, gr in zip(qwen_rows, mistral_rows, granite_rows):
        assert qr["context"] == mr["context"] == gr["context"], "row order mismatch across qwen/mistral/granite CSVs"
        r = dict(qr)
        if qr["category"] != "word":
            rows.append((r, None, None))
            continue
        qwen_scores = dict(parse_preds(qr["tf_preds"]))
        mistral_scores = dict(parse_preds(mr["tf_preds"]))
        granite_scores = dict(parse_preds(gr["tf_preds"]))
        ngram_best = ngram_best_of(r)
        candidates = set(qwen_scores) | set(mistral_scores) | set(granite_scores) | ({ngram_best} if ngram_best else set())
        ctx_tokens = r["context"].split()
        quads = {w: (qwen_scores.get(w, FLOOR), mistral_scores.get(w, FLOOR),
                     granite_scores.get(w, FLOOR), kn_scorer.score(ctx_tokens, w))
                 for w in candidates}
        rows.append((r, quads, ngram_best))
    print(f"done precomputing ({len(rows)} rows)", flush=True)

    grid = build_grid(args.lam_steps)
    folds = stratified_kfold(rows, args.folds, args.split_seed, category_of=lambda row: row[0]["category"])
    print(f"\n{args.folds}-fold CV, {len(rows)} rows, {len(grid)} (lam_q, lam_m, lam_g) grid points")

    cat_correct, cat_total = {}, {}
    chosen = []
    for i in range(args.folds):
        test_fold = folds[i]
        train_rows = [row for j, f in enumerate(folds) if j != i for row in f]

        best_lams, best_acc = best_lambda(train_rows, grid)
        chosen.append(best_lams)

        for r, quads, ngram_best in test_fold:
            cat = r["category"]
            cat_total[cat] = cat_total.get(cat, 0) + 1
            if cat == "symbol":
                pred = r["first_letter"]
            elif cat == "number":
                pred = ("1" * pick_length(number_scorer, r["context"].split())
                        if number_scorer else ngram_best_of(r))
            else:
                pred = pick_best(quads, *best_lams)
            ok = (pred or "").strip().lower() == r["answer"].strip().lower()
            cat_correct[cat] = cat_correct.get(cat, 0) + ok
        print(f"fold {i+1}/{args.folds}: lam_qwen={best_lams[0]:.2f} lam_mistral={best_lams[1]:.2f} "
              f"lam_granite={best_lams[2]:.2f} (train word acc {best_acc*100:.2f}%)", flush=True)

    print(f"\nper-fold chosen (lam_qwen, lam_mistral, lam_granite): "
          f"{[(f'{q:.2f}', f'{m:.2f}', f'{g:.2f}') for q, m, g in chosen]}")
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
