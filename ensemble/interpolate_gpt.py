#!/usr/bin/env python3
"""Part 2 extended: 3-way interpolation of Model A, Model B (KenLM) and the GPT2 transformer --
but scored on the TRUE ANSWER's probability (next-word setting), not the whole context sentence
interpolate.py uses. Fast proxy for whether blending GPT with the n-grams helps at all: log10
P(true answer | context) under each of the three models, blended in probability space, perplexity
minimized by grid search -- same shape and speed class as interpolate.py (one kenlm state-prime +
one BaseFullScore call per row per n-gram model; GPT's score is already sitting in
gpt/33k_full_gpt/devv_*_predictions.csv's gpt_logprob10 column, no rescoring there).

NOT the same number as blended top-1 accuracy (ensemble_gpt_ngram.py does that -- ~4x slower
than ngram/predict_accuracy.py's own 2-model interpolated run because it needs a fresh per-candidate
kenlm score for every same-first-letter word, not just the true answer). Likelihood mass on the
truth correlates with accuracy but isn't literally it -- use this as a quick "is there anything
to gain by combining GPT with the n-grams" check before paying for the real (slow) one.

Alnum rows only (mirrors report.md's alnum-only convention) -- symbol rows are answered
deterministically, no model involved, nothing to interpolate. Reads context/first letter/answer
straight from the GPT predictions csv (same devv_eval.csv/devv_test.csv, same row order) --
no need to also open the raw data csv.

Usage:
    python3 ensemble/interpolate_gpt.py
"""
import argparse
import math
import sys
from pathlib import Path
import csv

import kenlm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ngram"))
sys.path.insert(0, str(ROOT))

from predict_accuracy import _prime
from scripts.symbol_predict import is_symbol_letter

RUN_DIR = ROOT / "weights" / "run_20260905_103654"
GPT_DIR = ROOT / "gpt" / "33k_full_gpt"


def load_rows(gpt_preds_csv, model_a, model_b):
    """Returns list of (n_words, log_a, log_b, log_gpt) for alnum rows only."""
    with open(gpt_preds_csv, newline="", encoding="utf-8") as f:
        gpt_rows = list(csv.DictReader(f))

    rows = []
    s1, s2 = kenlm.State(), kenlm.State()
    for row in gpt_rows:
        letter = row["first letter"]
        if is_symbol_letter(letter) or row["gpt_logprob10"] in (None, ""):
            continue
        tokens, answer = row["context"].split(), row["answer"]
        n_words = len(tokens) + 1

        state_a = _prime(model_a, tokens, s1, s2)
        log_a = model_a.BaseFullScore(state_a, answer, kenlm.State()).log_prob

        state_b = _prime(model_b, tokens, s1, s2)
        log_b = model_b.BaseFullScore(state_b, answer, kenlm.State()).log_prob

        rows.append((n_words, log_a, log_b, float(row["gpt_logprob10"])))
    return rows


def perplexity(rows, w_a, w_b, w_g):
    total_logprob = total_words = 0.0
    for n_words, log_a, log_b, log_g in rows:
        p = w_a * (10 ** log_a) + w_b * (10 ** log_b) + w_g * (10 ** log_g)
        log_p = math.log10(p) if p > 0 else -300.0
        total_logprob += log_p
        total_words += n_words
    return 10 ** (-total_logprob / total_words)


def grid_search(rows, step=0.05):
    best, best_ppl = None, float("inf")
    n_steps = round(1.0 / step)
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            w_a, w_b = i * step, j * step
            w_g = 1.0 - w_a - w_b
            if w_g < -1e-9:
                continue
            ppl = perplexity(rows, w_a, w_b, w_g)
            if ppl < best_ppl:
                best, best_ppl = (w_a, w_b, w_g), ppl
    return best, best_ppl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-gpt-preds", type=Path, default=GPT_DIR / "devv_eval_predictions.csv")
    ap.add_argument("--test-gpt-preds", type=Path, default=GPT_DIR / "devv_test_predictions.csv")
    ap.add_argument("--model-a", type=Path, default=RUN_DIR / "model_a.klm")
    ap.add_argument("--model-b", type=Path, default=RUN_DIR / "model_b.klm")
    ap.add_argument("--step", type=float, default=0.05, help="weight grid step")
    args = ap.parse_args()

    model_a = kenlm.Model(str(args.model_a))
    model_b = kenlm.Model(str(args.model_b))

    eval_rows = load_rows(args.eval_gpt_preds, model_a, model_b)
    test_rows = load_rows(args.test_gpt_preds, model_a, model_b)
    print(f"eval rows: {len(eval_rows)}, test rows: {len(test_rows)}")

    print(f"eval perplexity -- model A alone: {perplexity(eval_rows, 1, 0, 0):.3f}")
    print(f"eval perplexity -- model B alone: {perplexity(eval_rows, 0, 1, 0):.3f}")
    print(f"eval perplexity -- GPT alone:     {perplexity(eval_rows, 0, 0, 1):.3f}")

    (w_a, w_b, w_g), best_ppl = grid_search(eval_rows, args.step)
    print(f"\nbest weights (tuned on devv_eval, step {args.step}): "
          f"model_a={w_a:.2f} model_b={w_b:.2f} gpt={w_g:.2f} -> eval perplexity {best_ppl:.3f}")

    print("\napplied to devv_test:")
    print(f"  model A alone: {perplexity(test_rows, 1, 0, 0):.3f}")
    print(f"  model B alone: {perplexity(test_rows, 0, 1, 0):.3f}")
    print(f"  GPT alone:     {perplexity(test_rows, 0, 0, 1):.3f}")
    print(f"  interpolated (a={w_a:.2f}, b={w_b:.2f}, gpt={w_g:.2f}): {perplexity(test_rows, w_a, w_b, w_g):.3f}")


if __name__ == "__main__":
    main()
