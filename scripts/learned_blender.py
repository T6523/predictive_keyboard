"""Learned blender (status.md advice item 2): replace the scalar lambda blend with
a small GBDT (sklearn.HistGradientBoostingClassifier -- already installed, no new
dependency) that predicts P(candidate is correct) from features. Point is to hand
the model genuinely NEW information the LM's own log-prob doesn't carry -- n-gram
backoff level and KN5 log-prob -- not just rescore the LM's own signal (that's the
failure mode that killed the earlier MiniLM reranker, see report.md: "no new
information available to a second model conditioned on the same left context").

Candidate set per row = whatever's already in qwen_tf_scores.csv's tf_preds column
(LM beam top5 union n-gram top10, per-candidate teacher-forced log-prob -- see
score_candidates.py). Pointwise-to-listwise: train on (row, candidate) -> is_answer,
pick argmax P(correct) per row at inference.

Usage:
    python3 learned_blender.py --data ../weights/qwen_tf_scores.csv \
        --ngram ../weights/ngram_4_a.bin --kn-model ../weights/kn5.binary
"""
import argparse
import csv
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from blend_ngram import KN5Scorer, _predict_word, parse_preds, stratified_kfold
from infer_ngram import load_model

FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "at", "for",
    "from", "that", "this", "these", "those", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "shall", "should", "may", "might", "must", "not", "with",
    "as", "by", "it", "he", "she", "they", "we", "you", "i", "their", "his",
    "her", "its", "our", "your", "my",
}

FLOOR = math.log(1e-10)


def backoff_level(context_tokens, candidate, n, counts, vocab):
    """Highest n-gram order j (1..n-1) where candidate was actually seen after
    this exact context, 0 if only unigram or never seen -- info the LM's own
    log-prob doesn't carry."""
    cand_id = vocab.get(candidate.lower())
    if cand_id is None:
        return 0
    ids = [vocab.get(t) for t in context_tokens[-(n - 1):]]
    for j in range(len(ids), 0, -1):
        ctx_ids = ids[-j:]
        if None in ctx_ids:
            continue
        d = counts[j].get(tuple(ctx_ids))
        if d and d.get(cand_id, 0) > 0:
            return j
    return 0


def row_features(context, candidates, tf_scores, kn_scorer, n, counts, vocab):
    """One feature vector per candidate word."""
    ctx_tokens = context.split()
    uni = counts[0][()]
    ranked = sorted(candidates, key=lambda w: -tf_scores.get(w, FLOOR))
    rank_of = {w: i for i, w in enumerate(ranked)}
    feats = []
    for w in candidates:
        cand_id = vocab.get(w.lower())
        uni_c = uni.get(cand_id, 0) if cand_id is not None else 0
        feats.append([
            tf_scores.get(w, FLOOR),
            rank_of[w],
            kn_scorer.score(ctx_tokens, w),
            backoff_level(ctx_tokens, w, n, counts, vocab),
            math.log(uni_c + 1),
            len(w),
            1.0 if w.lower() in FUNCTION_WORDS else 0.0,
        ])
    return feats


def demo():
    """Self-check: backoff_level backoff-order logic, no model file needed."""
    vocab = {"cat": 0, "car": 1, "dog": 2}
    counts = {0: {(): {0: 5, 1: 3, 2: 10}}, 1: {(2,): {0: 1}}}  # after "dog": only cat seen
    assert backoff_level(["dog"], "cat", 2, counts, vocab) == 1  # order-1 hit
    assert backoff_level(["dog"], "car", 2, counts, vocab) == 0  # only unigram-level
    assert backoff_level(["dog"], "xyz", 2, counts, vocab) == 0  # unseen -> OOV, 0
    print("demo ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../weights/qwen_tf_scores.csv")
    ap.add_argument("--ngram", default="../weights/ngram_4_a.bin")
    ap.add_argument("--kn-model", default="../weights/kn5.binary")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    n, counts, vocab, id_to_tok = load_model(args.ngram)
    kn_scorer = KN5Scorer(args.kn_model)
    rows = list(csv.DictReader(open(args.data, encoding="utf-8")))
    word_rows = [r for r in rows if r["category"] == "word"]

    # precompute per-row candidates/features once -- reused across all 5 folds
    print(f"building features for {len(word_rows)} word rows...", flush=True)
    row_data = []  # (row, candidates, feats, answer_idx or -1)
    for r in word_rows:
        tf_scores = dict(parse_preds(r["tf_preds"]))
        candidates = list(tf_scores)
        feats = row_features(r["context"], candidates, tf_scores, kn_scorer, n, counts, vocab)
        ans = r["answer"].strip().lower()
        ans_idx = next((i for i, w in enumerate(candidates) if w.lower() == ans), -1)
        row_data.append((r, candidates, feats, ans_idx))

    folds = stratified_kfold(word_rows, args.folds, args.split_seed)
    fold_of_row = {id(r): i for i, f in enumerate(folds) for r in f}

    correct = total = 0
    for i in range(args.folds):
        train = [(cands, feats, ans_idx) for r, cands, feats, ans_idx in row_data if fold_of_row[id(r)] != i]
        test = [(r, cands, feats, ans_idx) for r, cands, feats, ans_idx in row_data if fold_of_row[id(r)] == i]

        X_train, y_train = [], []
        for cands, feats, ans_idx in train:
            for j, f in enumerate(feats):
                X_train.append(f)
                y_train.append(1 if j == ans_idx else 0)

        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0)
        clf.fit(np.array(X_train), np.array(y_train))

        for r, cands, feats, ans_idx in test:
            proba = clf.predict_proba(np.array(feats))[:, 1]
            pred_idx = int(np.argmax(proba))
            correct += pred_idx == ans_idx
            total += 1
        print(f"fold {i+1}/{args.folds} done", flush=True)

    print(f"\n--- learned blender, pooled {args.folds}-fold CV, word category ---")
    print(f"word {correct}/{total} = {correct/total*100:.2f}%")

    # full routing accuracy for comparison (symbol + number unchanged from blend_ngram.py)
    other_rows = [r for r in rows if r["category"] != "word"]
    sym_c = sym_t = num_c = num_t = 0
    for r in other_rows:
        if r["category"] == "symbol":
            sym_t += 1
            sym_c += r["first_letter"].strip().lower() == r["answer"].strip().lower()
        elif r["category"] == "number":
            num_t += 1
            pred = _predict_word(r["context"].split(), r["first_letter"], n, counts, vocab, id_to_tok)
            num_c += (pred or "").strip().lower() == r["answer"].strip().lower()
    tot_c, tot_n = correct + sym_c + num_c, total + sym_t + num_t
    print(f"symbol {sym_c}/{sym_t} = {sym_c/sym_t*100:.2f}%" if sym_t else "symbol 0/0")
    print(f"number {num_c}/{num_t} = {num_c/num_t*100:.2f}%" if num_t else "number 0/0")
    print(f"overall {tot_c}/{tot_n} = {tot_c/tot_n*100:.2f}%")


if __name__ == "__main__":
    main()
