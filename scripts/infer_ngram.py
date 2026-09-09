"""Run the trained ngram_N.bin model on dev_set_final.csv, predict the answer word given
context + first-letter constraint, and report accuracy split by answer category (word /
number / symbol). Symbol category is predicted trivially as the given first letter itself
(see dev_eda.ipynb -- majority-per-first-letter is ~99.6% for symbols already).

Usage:
    python3 infer_ngram.py --model ../weights/ngram_4_a.bin --dev ../data/dev_set_final.csv \
        --out ../weights/dev_predictions_a.csv
"""
import argparse
import csv
import pickle
import re
from collections import defaultdict

CAT_NUMBER = re.compile(r"[0-9]+")
CAT_WORD = re.compile(r"[a-z']+", re.IGNORECASE)


def categorize(tok):
    if CAT_NUMBER.fullmatch(tok):
        return "number"
    if CAT_WORD.fullmatch(tok):
        return "word"
    return "symbol"


def best_by_letter(d, letter, id_to_tok):
    """Argmax count in a (word_id -> count) dict, restricted to id_to_tok[id][0] == letter."""
    best_id, best_c = None, -1
    for wid, c in d.items():
        tok = id_to_tok[wid]
        if tok and tok[0] == letter and c > best_c:
            best_id, best_c = wid, c
    return best_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../weights/ngram_4_a.bin")
    ap.add_argument("--dev", default="../data/dev_set_final.csv")
    ap.add_argument("--out", default="../weights/dev_predictions_a.csv")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        m = pickle.load(f)
    n, counts, vocab, id_to_tok = m["n"], m["counts"], m["vocab"], m["id_to_tok"]

    # unigram fallback: best word per starting letter, precomputed once
    letter_ids = defaultdict(list)
    for tok, i in vocab.items():
        if tok in ("<s>", "</s>") or not tok:
            continue
        letter_ids[tok[0]].append(i)
    unigram = counts[0][()]
    unigram_best = {
        letter: max(ids, key=lambda i: unigram.get(i, 0)) for letter, ids in letter_ids.items()
    }

    def predict_word(context_tokens, letter):
        ids = [vocab.get(t) for t in context_tokens[-(n - 1):]]
        for j in range(len(ids), 0, -1):
            ctx_ids = ids[-j:]
            if None in ctx_ids:
                continue
            d = counts[j].get(tuple(ctx_ids))
            if d:
                cand = best_by_letter(d, letter, id_to_tok)
                if cand is not None:
                    return id_to_tok[cand]
        cand = unigram_best.get(letter)
        return id_to_tok[cand] if cand is not None else ""

    rows_out = []
    correct = defaultdict(int)
    total = defaultdict(int)
    with open(args.dev, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            answer, letter, context = row["answer"], row["first letter"], row["context"]
            cat = categorize(answer)
            if cat == "symbol":
                pred = letter  # symbols: just predict the given first char
            else:
                pred = predict_word(context.split(), letter)
            ok = pred == answer
            correct[cat] += ok
            total[cat] += 1
            rows_out.append({
                "context": context, "first letter": letter, "answer": answer,
                "answer_cat": cat, "prediction": pred, "correct": int(ok),
            })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    print(f"predictions -> {args.out}")
    overall_c, overall_t = sum(correct.values()), sum(total.values())
    print(f"overall: {overall_c}/{overall_t} = {overall_c/overall_t:.4f}")
    for cat in ("word", "symbol", "number"):
        c, t = correct[cat], total[cat]
        print(f"{cat}: {c}/{t} = {c/t:.4f}" if t else f"{cat}: 0/0")


if __name__ == "__main__":
    main()
