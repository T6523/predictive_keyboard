"""Error EDA on blend3.py's actual mistakes (post-blend, current 74.64% word acc) --
NOT the old pre-blend LoRA-only errors. Same pooled 5-fold CV harness as blend3.py,
but dumps every wrong word-row (context, answer, pred, top-3 candidates by score)
instead of just the accuracy number.

Usage:
    python3 error_eda.py --out /tmp/claude-1000/.../scratchpad/blend3_misses.csv
"""
import argparse
import csv
from collections import Counter

from blend3 import pick_best
from blend_ngram import FLOOR, KN5Scorer, _predict_word, parse_preds, stratified_kfold
from infer_ngram import load_model


FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "at", "for", "from",
    "that", "this", "these", "those", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could", "shall",
    "should", "may", "might", "must", "not", "with", "as", "by", "it", "he", "she",
    "they", "we", "you", "i", "their", "his", "her", "its", "our", "your", "my",
}


def demo():
    """Self-check: classify() buckets a few hand-picked cases as expected."""
    assert classify("to", "the") == "function_word"
    assert classify("dog", "dogs") == "morphology"
    assert classify("hous", "house") == "prefix_truncation"
    assert classify("xyz", "abc") == "other"
    print("demo ok")


def classify(pred, ans):
    pred, ans = (pred or "").lower(), ans.lower()
    if pred in FUNCTION_WORDS and ans in FUNCTION_WORDS:
        return "function_word"
    # cheap morphology check: same stem +/- common suffix (dog/dogs, walk/walked)
    for suf in ("s", "es", "ed", "ing", "d", "er", "est", "ly"):
        if pred + suf == ans or ans + suf == pred:
            return "morphology"
    if ans.startswith(pred) or pred.startswith(ans):
        return "prefix_truncation"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", default="../weights/qwen_tf_scores.csv")
    ap.add_argument("--mistral", default="../weights/mistral_tf_scores.csv")
    ap.add_argument("--ngram", default="../weights/ngram_4_a.bin")
    ap.add_argument("--kn-model", default="../weights/kn5.binary")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lam-steps", type=int, default=11)
    ap.add_argument("--out", default="/tmp/blend3_misses.csv")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    n, counts, vocab, id_to_tok = load_model(args.ngram)
    kn_scorer = KN5Scorer(args.kn_model)

    qwen_rows = list(csv.DictReader(open(args.qwen, encoding="utf-8")))
    mistral_rows = list(csv.DictReader(open(args.mistral, encoding="utf-8")))

    def ngram_best_of(r):
        return _predict_word(r["context"].split(), r["first_letter"], n, counts, vocab, id_to_tok)

    print("precomputing triples...", flush=True)
    rows = []
    for qr, mr in zip(qwen_rows, mistral_rows):
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
    print("done", flush=True)

    grid = [(i / (args.lam_steps - 1), j / (args.lam_steps - 1))
            for i in range(args.lam_steps) for j in range(args.lam_steps - i)]
    folds = stratified_kfold(rows, args.folds, args.split_seed, category_of=lambda row: row[0]["category"])

    def eval_word(split_rows, lam_q, lam_m):
        correct = total = 0
        for r, triples, _ in split_rows:
            if r["category"] != "word":
                continue
            pred = pick_best(triples, lam_q, lam_m)
            correct += (pred or "").strip().lower() == r["answer"].strip().lower()
            total += 1
        return correct, total

    misses = []
    for i in range(args.folds):
        test_fold = folds[i]
        train_rows = [row for j, f in enumerate(folds) if j != i for row in f]
        best_lams, best_acc = (1.0, 0.0), -1
        for lam_q, lam_m in grid:
            c, t = eval_word(train_rows, lam_q, lam_m)
            acc = c / t if t else 0
            if acc > best_acc:
                best_lams, best_acc = (lam_q, lam_m), acc

        for r, triples, ngram_best in test_fold:
            if r["category"] != "word":
                continue
            pred = pick_best(triples, *best_lams)
            ans = r["answer"].strip().lower()
            if (pred or "").strip().lower() != ans:
                ranked = sorted(triples.items(),
                                 key=lambda kv: best_lams[0] * kv[1][0] + best_lams[1] * kv[1][1]
                                 + (1 - best_lams[0] - best_lams[1]) * kv[1][2],
                                 reverse=True)[:3]
                misses.append({
                    "context": r["context"],
                    "first_letter": r["first_letter"],
                    "answer": r["answer"],
                    "pred": pred,
                    "ans_in_candidates": ans in triples,
                    "top3": "; ".join(f"{w}" for w, _ in ranked),
                    "err_type": classify(pred, ans),
                })

    print(f"\n{len(misses)} word misses out of 8198\n")
    counts_by_type = Counter(m["err_type"] for m in misses)
    for t, c in counts_by_type.most_common():
        print(f"{t:20s} {c:5d}  {c/len(misses)*100:.1f}%")

    in_cand = sum(m["ans_in_candidates"] for m in misses)
    print(f"\nanswer was in candidate set but lost the argmax: {in_cand}/{len(misses)} "
          f"({in_cand/len(misses)*100:.1f}%)")
    print(f"answer never made it into candidate set (ceiling loss): {len(misses)-in_cand}/{len(misses)} "
          f"({(len(misses)-in_cand)/len(misses)*100:.1f}%)")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["context", "first_letter", "answer", "pred",
                                           "ans_in_candidates", "top3", "err_type"])
        w.writeheader()
        w.writerows(misses)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
