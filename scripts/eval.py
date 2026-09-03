"""End-to-end eval: symbol/[UNK] first letters are answered deterministically, everything else
goes through a model (default: the n-gram from train_ngram.py). Scores all rows together.

Takes any csv with 'context' + 'first letter' columns; 'answer' is optional -- if present,
accuracy is scored (overall + broken down by route), if absent predictions are just written out.

Usage:
    python3 eval.py --data ../data/devv_test.csv --model ../weights/ngram_3.bin
    python3 eval.py --data ../data/test_set_no_answer.csv --model ../weights/ngram_3.bin --out preds.csv
"""
import argparse
import csv
import time

from symbol_predict import is_symbol_letter, predict_symbol
from eval_ngram import load_model, predict as ngram_predict


def clean_tokens(context):
    return [t for t in context.lower().split() if t.isalnum()]


def predict_row(context, letter, model):
    if is_symbol_letter(letter):
        return predict_symbol(letter), "symbol"
    pred, _ = ngram_predict(model, clean_tokens(context), letter)
    return pred, "ngram"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="../weights/ngram_3.bin")
    ap.add_argument("--out", default=None, help="write predictions to this csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model = load_model(args.model)
    print(f"loaded {model['n']}-gram model from {args.model}")

    with open(args.data, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    has_answer = rows and "answer" in rows[0]

    t0 = time.time()
    preds, routes = [], []
    correct = correct_by_route = total_by_route = None
    if has_answer:
        correct, correct_by_route, total_by_route = 0, {"symbol": 0, "ngram": 0}, {"symbol": 0, "ngram": 0}

    for row in rows:
        pred, route = predict_row(row["context"], row["first letter"], model)
        preds.append(pred)
        routes.append(route)
        if has_answer:
            total_by_route[route] += 1
            if pred == row["answer"]:
                correct += 1
                correct_by_route[route] += 1

    n = len(rows)
    print(f"rows: {n}, eval time: {time.time()-t0:.1f}s")
    print(f"routed: symbol={routes.count('symbol')}, ngram={routes.count('ngram')}")

    if has_answer:
        print(f"overall accuracy: {correct}/{n} = {correct/n:.4f}")
        for route in ("symbol", "ngram"):
            t = total_by_route[route]
            if t:
                print(f"  {route}: {correct_by_route[route]}/{t} = {correct_by_route[route]/t:.4f}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["prediction", "route"])
            w.writerows(zip(preds, routes))
        print(f"predictions -> {args.out}")


if __name__ == "__main__":
    main()
