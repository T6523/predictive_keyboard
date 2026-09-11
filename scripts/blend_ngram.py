"""Blend the fine-tuned Qwen LoRA (or zero-shot, --lm-source) with the local n-gram
(TODO.md Suggestion item 5), on the matched-eval CSV exported from Kaggle
(kaggle/matched_eval/, columns: context, first_letter, answer, category,
zeroshot_preds, lora_preds -- "word:score; word:score; ...").

Production routing (per this session's request):
    symbol -> deterministic letter copy (already ~99% ceiling)
    word   -> blend: score(w) = lambda*lm_logprob(w) + (1-lambda)*ngram_logprob(w),
              candidate set = LM's top5 union the n-gram's own best guess (so the
              blend can recover a word the LM's beam never generated, not just
              rerank the LM's list)
    number -> entirely n-gram (infer_ngram.py already gets 71.68% there by
              conditioning digit-length on context; the LM structurally can't
              produce the anonymized 1111-style placeholder, see report.md)

Stratified 80/20 split (by category, seed --split-seed): lambda is grid-searched on
the 80% tune split, accuracy reported on the untouched 20% holdout.

Usage:
    python3 blend_ngram.py --data ../weights/qwen_matched_scores.csv \
        --ngram ../weights/ngram_4_a.bin --lm-source lora
"""
import argparse
import csv
import math
import random

import kenlm

from infer_ngram import load_model

LN10 = math.log(10)


class KN5Scorer:
    """Wraps a KenLM modified-Kneser-Ney model, replacing the old unsmoothed
    count-based score_word() as the blend's n-gram-side probability source
    (status.md advice item 1 -- "cheapest, likely +1-2pt")."""

    def __init__(self, path, order=5):
        self.model = kenlm.Model(path)
        self.order = order

    def score(self, context_tokens, candidate):
        ctx = context_tokens[-(self.order - 1):]
        text = " ".join(ctx + [candidate.lower()])
        log10p, _, _ = list(self.model.full_scores(text, bos=False, eos=False))[-1]
        return log10p * LN10  # natural log, matches score_word's convention


def parse_preds(field):
    """"word:score; word:score; ..." -> [(word, score), ...], [] if empty."""
    if not field:
        return []
    out = []
    for part in field.split("; "):
        word, _, score = part.rpartition(":")
        out.append((word, float(score)))
    return out


FLOOR = math.log(1e-10)  # same smoothing floor as score_word's own unseen case --
                          # used on whichever side (lm/ngram) doesn't cover a candidate


def blend_candidates(lm_preds, ngram_best, context, kn_scorer, lam):
    """Candidate set = LM's top5 union the n-gram's own top pick. Returns the
    highest blended-score candidate."""
    lm_scores = dict(lm_preds)
    candidates = set(lm_scores) | ({ngram_best} if ngram_best else set())
    best_word, best_score = None, float("-inf")
    for w in candidates:
        lm_s = lm_scores.get(w, FLOOR)
        ng_s = kn_scorer.score(context.split(), w)
        s = lam * lm_s + (1 - lam) * ng_s
        if s > best_score:
            best_word, best_score = w, s
    return best_word


def stratified_kfold(rows, k, seed, category_of=lambda r: r["category"]):
    """k folds, stratified by category. A single 80/20 split leaves only ~1890
    holdout rows (+-2pt noise at 95% CI, per status.md's own advice) -- pooling
    predictions across k folds evaluates all N rows, each held out exactly once.
    category_of lets callers pass non-dict rows (blend3.py's precomputed tuples)."""
    rng = random.Random(seed)
    by_cat = {}
    for r in rows:
        by_cat.setdefault(category_of(r), []).append(r)
    folds = [[] for _ in range(k)]
    for cat, group in by_cat.items():
        group = group[:]
        rng.shuffle(group)
        for i, r in enumerate(group):
            folds[i % k].append(r)
    return folds


def demo():
    """Self-check: pred parsing + candidate union + split proportions, no model/network."""
    assert parse_preds("") == []
    assert parse_preds("cat:-0.5000; dog:-1.2000") == [("cat", -0.5), ("dog", -1.2)]

    rows = [{"category": "word"}] * 80 + [{"category": "number"}] * 20
    folds = stratified_kfold(rows, 5, seed=0)
    assert sum(len(f) for f in folds) == 100
    for f in folds:
        assert 14 <= sum(r["category"] == "word" for r in f) <= 18  # ~16/fold, proportional
    print("demo ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../weights/qwen_matched_scores.csv")
    ap.add_argument("--ngram", default="../weights/ngram_4_a.bin")
    ap.add_argument("--kn-model", default="../weights/kn5.binary")
    ap.add_argument("--lm-source", choices=["zeroshot", "lora", "tf"], default="lora")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lam-steps", type=int, default=21, help="grid points from 0.0 to 1.0")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    n, counts, vocab, id_to_tok = load_model(args.ngram)
    kn_scorer = KN5Scorer(args.kn_model)
    rows = list(csv.DictReader(open(args.data, encoding="utf-8")))
    pred_col = f"{args.lm_source}_preds"

    # matched zero-shot vs lora comparison, straight from the exported CSV -- both
    # already ran the same word rows, same beam-k5 protocol on Kaggle (skipped for
    # qwen_tf_scores.csv, which only carries the "tf" column)
    word_rows = [r for r in rows if r["category"] == "word"]
    for col, label in [("zeroshot_preds", "zero-shot"), ("lora_preds", "lora")]:
        if col not in word_rows[0]:
            continue
        c1 = c5 = 0
        for r in word_rows:
            cands = [w.lower() for w, _ in parse_preds(r[col])]
            ans = r["answer"].strip().lower()
            c1 += cands[:1] == [ans]
            c5 += ans in cands[:5]
        print(f"{label:10s} top1 {c1}/{len(word_rows)} = {c1/len(word_rows)*100:.2f}%  "
              f"top5 {c5}/{len(word_rows)} = {c5/len(word_rows)*100:.2f}%")

    folds = stratified_kfold(rows, args.folds, args.split_seed)
    print(f"\n{args.folds}-fold CV, {len(rows)} rows total (lm-source={args.lm_source})")

    def ngram_best_of(r):
        return _predict_word(r["context"].split(), r["first_letter"], n, counts, vocab, id_to_tok)

    def eval_word(split_rows, lam):
        correct = total = 0
        for r in split_rows:
            if r["category"] != "word":
                continue
            pred = blend_candidates(parse_preds(r[pred_col]), ngram_best_of(r), r["context"], kn_scorer, lam)
            correct += (pred or "").strip().lower() == r["answer"].strip().lower()
            total += 1
        return correct, total

    # pooled k-fold CV: each fold held out once, lambda tuned on the other k-1 folds.
    # Every row gets exactly one out-of-fold prediction -- avoids the +-2pt noise of a
    # single 80/20 split on n=1890 (status.md advice).
    cat_correct, cat_total = {}, {}
    lambdas = []
    for i in range(args.folds):
        test_fold = folds[i]
        train_rows = [r for j, f in enumerate(folds) if j != i for r in f]

        best_lam, best_acc = 0.5, -1
        for step in range(args.lam_steps):
            lam = step / (args.lam_steps - 1)
            c, t = eval_word(train_rows, lam)
            acc = c / t if t else 0
            if acc > best_acc:
                best_lam, best_acc = lam, acc
        lambdas.append(best_lam)

        for r in test_fold:
            cat = r["category"]
            cat_total[cat] = cat_total.get(cat, 0) + 1
            if cat == "symbol":
                pred = r["first_letter"]
            elif cat == "number":
                pred = ngram_best_of(r)
            else:  # word
                pred = blend_candidates(parse_preds(r[pred_col]), ngram_best_of(r), r["context"], kn_scorer, best_lam)
            ok = (pred or "").strip().lower() == r["answer"].strip().lower()
            cat_correct[cat] = cat_correct.get(cat, 0) + ok

    print(f"per-fold tuned lambda: {[f'{l:.2f}' for l in lambdas]}")
    print(f"\n--- pooled CV ({sum(cat_total.values())} rows) ---")
    tot_c = tot_n = 0
    for cat in ("word", "symbol", "number"):
        c, t = cat_correct.get(cat, 0), cat_total.get(cat, 0)
        tot_c += c
        tot_n += t
        if t:
            print(f"{cat:8s} {c:5d}/{t:5d}  {c/t*100:.2f}%")
    print(f"{'overall':8s} {tot_c:5d}/{tot_n:5d}  {tot_c/tot_n*100:.2f}%")


def _predict_word(context_tokens, letter, n, counts, vocab, id_to_tok):
    """Same backoff argmax as infer_ngram.py's predict_word (that one's a closure
    over an open file, can't import directly) -- used for both the number-category
    track and as the blend's n-gram-side candidate."""
    from collections import defaultdict

    def best_by_letter(d, letter):
        best_id, best_c = None, -1
        for wid, c in d.items():
            tok = id_to_tok[wid]
            if tok and tok[0] == letter and c > best_c:
                best_id, best_c = wid, c
        return best_id

    ids = [vocab.get(t) for t in context_tokens[-(n - 1):]]
    for j in range(len(ids), 0, -1):
        ctx_ids = ids[-j:]
        if None in ctx_ids:
            continue
        d = counts[j].get(tuple(ctx_ids))
        if d:
            cand = best_by_letter(d, letter)
            if cand is not None:
                return id_to_tok[cand]
    cand = best_by_letter(counts[0][()], letter)
    return id_to_tok[cand] if cand is not None else ""


if __name__ == "__main__":
    main()
