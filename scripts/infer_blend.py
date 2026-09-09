"""Linear-interpolation blend of model A (ngram_4_a.bin, trained on train_final) and
model B (ngram_4_b.bin, trained on masked gigaword): P_blend(w) = lambda*P_A(w) +
(1-lambda)*P_B(w), argmax over the union of both models' first-letter-matching
candidates. P_A/P_B are relative frequencies at whichever backoff order each model
resolves to (count(w) / sum(all counts in that context), not raw counts -- the two
models have very different corpus sizes so raw counts aren't comparable).

dev_set_final.csv is split 80/20 (fixed seed): lambda is grid-searched on the 20% tune
split, then the picked lambda is scored once on the 80% held-out split. Symbol category
is untouched (both models already use the same trivial "predict the given letter" rule,
independent of lambda).

Usage:
    python3 infer_blend.py
"""
import csv
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
WEIGHTS = Path(__file__).resolve().parent.parent / "weights"

CAT_NUMBER = re.compile(r"[0-9]+")
CAT_WORD = re.compile(r"[a-z']+", re.IGNORECASE)


def categorize(tok):
    if CAT_NUMBER.fullmatch(tok):
        return "number"
    if CAT_WORD.fullmatch(tok):
        return "word"
    return "symbol"


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def candidate_probs(m, context_tokens, letter):
    """Backs off order n-1 -> 0, returns {word: relative-freq} for the first order
    that has any candidate starting with `letter` (probs normalized within that
    context's full count, i.e. among ALL words, not just the letter-filtered ones)."""
    n, counts, vocab, id_to_tok = m["n"], m["counts"], m["vocab"], m["id_to_tok"]
    ids = [vocab.get(t) for t in context_tokens[-(n - 1):]]
    for j in range(len(ids), -1, -1):
        if j == 0:
            d = counts[0][()]
        else:
            ctx_ids = ids[-j:]
            if None in ctx_ids:
                continue
            d = counts[j].get(tuple(ctx_ids))
        if not d:
            continue
        total = sum(d.values())
        probs = {id_to_tok[w]: c / total for w, c in d.items() if id_to_tok[w][0:1] == letter}
        if probs:
            return probs
    return {}


def blend_predict(pa, pb, lam):
    words = pa.keys() | pb.keys()
    if not words:
        return ""
    return max(words, key=lambda w: lam * pa.get(w, 0.0) + (1 - lam) * pb.get(w, 0.0))


def main():
    model_a = load_model(WEIGHTS / "ngram_4_a.bin")
    model_b = load_model(WEIGHTS / "ngram_4_b.bin")

    rows = []
    with open(DATA / "dev_set_final.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    print(f"scoring {len(rows)} rows against both models (one pass, cached probs)...")
    scored = []
    for row in rows:
        answer, letter, context = row["answer"], row["first letter"], row["context"]
        cat = categorize(answer)
        ctx_toks = context.split()
        if cat == "symbol":
            scored.append((cat, answer, letter, None, None))  # symbol: rule-based, no model needed
        else:
            pa = candidate_probs(model_a, ctx_toks, letter)
            pb = candidate_probs(model_b, ctx_toks, letter)
            scored.append((cat, answer, letter, pa, pb))

    rng = random.Random(0)
    idx = list(range(len(scored)))
    rng.shuffle(idx)
    n_tune = int(len(idx) * 0.20)
    tune_idx, test_idx = idx[:n_tune], idx[n_tune:]
    print(f"split: {len(tune_idx)} tune (20%), {len(test_idx)} test (80%)")

    def accuracy(indices, lam):
        correct = 0
        for i in indices:
            cat, answer, letter, pa, pb = scored[i]
            pred = letter if cat == "symbol" else blend_predict(pa, pb, lam)
            correct += pred == answer
        return correct / len(indices)

    lambdas = [round(x * 0.05, 2) for x in range(21)]  # 0.00 .. 1.00
    tune_scores = [(lam, accuracy(tune_idx, lam)) for lam in lambdas]
    print("\nlambda sweep on tune split (20%):")
    for lam, acc in tune_scores:
        print(f"  lambda={lam:.2f}: acc={acc:.4f}")
    best_lam, best_tune_acc = max(tune_scores, key=lambda t: t[1])
    print(f"\nbest lambda from tune split: {best_lam} (tune acc={best_tune_acc:.4f})")

    def full_report(indices, lam, label):
        correct = defaultdict(int)
        total = defaultdict(int)
        for i in indices:
            cat, answer, letter, pa, pb = scored[i]
            pred = letter if cat == "symbol" else blend_predict(pa, pb, lam)
            correct[cat] += pred == answer
            total[cat] += 1
        print(f"\n{label} (n={len(indices)}, lambda={lam}):")
        overall_c, overall_t = sum(correct.values()), sum(total.values())
        print(f"  overall: {overall_c}/{overall_t} = {overall_c/overall_t:.4f}")
        for cat in ("word", "symbol", "number"):
            c, t = correct[cat], total[cat]
            print(f"  {cat}: {c}/{t} = {c/t:.4f}" if t else f"  {cat}: 0/0")

    full_report(test_idx, best_lam, "FINAL held-out result (80% test split)")

    # reference points: pure A (lambda=1) and pure B (lambda=0) on the same test split
    full_report(test_idx, 1.0, "reference: pure model A on test split")
    full_report(test_idx, 0.0, "reference: pure model B on test split")


if __name__ == "__main__":
    main()
