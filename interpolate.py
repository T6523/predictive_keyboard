#!/usr/bin/env python3
"""Part 2: interpolate Model A and Model B's whole-sentence log-probs.

Reuses model_a_logprob10 / model_b_logprob10 columns pipeline.py already wrote to
devv_eval_predictions.csv / devv_test_predictions.csv -- no model reload, no rescoring.

Interpolated prob per row: P = lambda*10^logA + (1-lambda)*10^logB (linear interpolation
in probability space, not log space -- standard LM interpolation).
Perplexity = 10^(-sum(log10 P) / total_word_count), word_count = context words + 1 (</s>).

Tune lambda on devv_eval (grid search, minimize eval perplexity), then apply that lambda
to devv_test and report test perplexity vs solo A / solo B.

Usage:
    python3 interpolate.py --eval-preds weights/run_.../devv_eval_predictions.csv \
                            --test-preds weights/run_.../devv_test_predictions.csv
"""
import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_rows(path):
    """Returns list of (word_count, log_a, log_b)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n_words = len(row["context"].split()) + 1  # +1 for </s>
            rows.append((n_words, float(row["model_a_logprob10"]), float(row["model_b_logprob10"])))
    return rows


def perplexity(rows, lam):
    """lam=1 -> pure model A, lam=0 -> pure model B."""
    total_logprob = 0.0
    total_words = 0
    for n_words, log_a, log_b in rows:
        if lam >= 1.0:
            log_p = log_a
        elif lam <= 0.0:
            log_p = log_b
        else:
            p = lam * (10 ** log_a) + (1 - lam) * (10 ** log_b)
            log_p = math.log10(p) if p > 0 else -300.0
        total_logprob += log_p
        total_words += n_words
    return 10 ** (-total_logprob / total_words)


def grid_search(rows, step=0.01):
    best_lam, best_ppl = None, float("inf")
    lam = 0.0
    while lam <= 1.0 + 1e-9:
        ppl = perplexity(rows, lam)
        if ppl < best_ppl:
            best_lam, best_ppl = lam, ppl
        lam += step
    return best_lam, best_ppl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-preds", required=True, type=Path)
    ap.add_argument("--test-preds", required=True, type=Path)
    ap.add_argument("--step", type=float, default=0.01, help="lambda grid step")
    args = ap.parse_args()

    eval_rows = load_rows(args.eval_preds)
    test_rows = load_rows(args.test_preds)

    print(f"eval rows: {len(eval_rows)}, test rows: {len(test_rows)}")
    print(f"eval perplexity -- model A alone (lambda=1): {perplexity(eval_rows, 1.0):.3f}")
    print(f"eval perplexity -- model B alone (lambda=0): {perplexity(eval_rows, 0.0):.3f}")

    best_lam, best_eval_ppl = grid_search(eval_rows, args.step)
    print(f"\nbest lambda (tuned on devv_eval): {best_lam:.2f} -> eval perplexity: {best_eval_ppl:.3f}")

    test_ppl_interp = perplexity(test_rows, best_lam)
    test_ppl_a = perplexity(test_rows, 1.0)
    test_ppl_b = perplexity(test_rows, 0.0)
    print(f"\napplied to devv_test:")
    print(f"  model A alone:        {test_ppl_a:.3f}")
    print(f"  model B alone:        {test_ppl_b:.3f}")
    print(f"  interpolated (λ={best_lam:.2f}): {test_ppl_interp:.3f}")


def _demo():
    """lambda=1 must reduce to pure A, lambda=0 to pure B; interpolation must beat the worse of A/B."""
    rows = [(3, -1.0, -5.0), (4, -2.0, -1.5), (2, -0.5, -0.5)]
    assert abs(perplexity(rows, 1.0) - perplexity([(n, a, a) for n, a, b in rows], 0.3)) < 1e-9
    lam, ppl = grid_search(rows, step=0.1)
    assert 0.0 <= lam <= 1.0
    assert ppl <= max(perplexity(rows, 1.0), perplexity(rows, 0.0)) + 1e-9
    print("self-test OK")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _demo()
    else:
        main()
