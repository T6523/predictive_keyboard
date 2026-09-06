#!/usr/bin/env python3
"""3-way next-word-accuracy ensemble: model_a (5-gram, in-domain), model_b (5-gram, gigaword),
and the from-scratch GPT2 checkpoint (gpt/33k_full_gpt/gpt2.pt) -- all three score every
same-first-letter candidate from weights/vocab.txt, argmax over a weighted probability blend,
weights grid-searched for best alnum accuracy on devv_eval then applied as-is to devv_test.
Same task/candidate pool as ngram/predict_accuracy.py (report.md §2), extended from 2 n-grams to 3
models. Symbol-letter rows untouched (deterministic rule, no model involved either way).

Per-row scoring (one kenlm BaseFullScore per candidate per n-gram model, one GPT forward pass
per row) is the expensive part -- done once and cached to disk, so the weight grid search itself
is pure dict arithmetic and reruns in seconds. Delete --cache's file to force a rescore (e.g.
after retraining any of the three models).

Usage:
    python3 ensemble/ensemble_gpt_ngram.py                        # full eval+test grid search
    python3 ensemble/ensemble_gpt_ngram.py --limit 500 --step 0.2  # quick smoke test, coarse grid
"""
import argparse
import pickle
import sys
import time
import csv
from pathlib import Path

import kenlm
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gpt" / "33k_full_gpt"))
sys.path.insert(0, str(ROOT / "ngram"))

from predict_accuracy import load_vocab_by_letter, _prime  # noqa: E402 -- reuse, don't duplicate
from infer import load_model as load_gpt, predict as gpt_predict  # noqa: E402

RUN_DIR = ROOT / "weights" / "run_20260905_103654"
LN10 = 2.302585092994046


def score_candidates_kenlm(model, context_tokens, candidates):
    """log10 P(word | context) for every candidate, from a context state primed once."""
    s1, s2 = kenlm.State(), kenlm.State()
    state = _prime(model, context_tokens, s1, s2)
    out = kenlm.State()
    return {w: model.BaseFullScore(state, w, out).log_prob for w in candidates}


def score_candidates_gpt(logits, gpt_vocab, candidates):
    """log10 P(word | context) for every candidate, sliced out of the one masked logit vector
    infer.py's predict() already computed for this row -- no second GPT forward pass."""
    logprobs = F.log_softmax(logits.float(), dim=-1)
    out = {}
    for w in candidates:
        wid = gpt_vocab.get(w)
        if wid is not None:
            out[w] = logprobs[wid].item() / LN10
    return out


def cache_row_scores(path, limit, model_a, model_b, gpt_model, gpt_vocab, id_to_tok, bucket_mask,
                      device, max_ctx, vocab_by_letter):
    from scripts.symbol_predict import is_symbol_letter

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]

    cached = []
    t0 = time.time()
    for n, row in enumerate(rows, 1):
        letter, answer = row["first letter"], row["answer"]
        if is_symbol_letter(letter):
            cached.append({"letter": letter, "answer": answer, "symbol": True})
        else:
            tokens = row["context"].split()
            candidates = vocab_by_letter.get(letter.lower(), ())
            if not candidates:
                cached.append({"letter": letter, "answer": answer, "symbol": False, "candidates": {}})
                continue
            log_a = score_candidates_kenlm(model_a, tokens, candidates)
            log_b = score_candidates_kenlm(model_b, tokens, candidates)
            _, logits = gpt_predict(gpt_model, gpt_vocab, id_to_tok, bucket_mask, device, tokens, letter, max_ctx)
            log_g = score_candidates_gpt(logits, gpt_vocab, candidates)
            per_word = {w: (log_a.get(w, -30.0), log_b.get(w, -30.0), log_g.get(w, -30.0)) for w in candidates}
            cached.append({"letter": letter, "answer": answer, "symbol": False, "candidates": per_word})
        if n % 2000 == 0:
            print(f"  scored {n}/{len(rows)} ({time.time()-t0:.0f}s)")
    print(f"{path.name}: scored {len(rows)} rows ({time.time()-t0:.0f}s)")
    return cached


def accuracy_at_weights(cached, w_a, w_b, w_g):
    from scripts.symbol_predict import predict_symbol

    correct = n_alnum = correct_alnum = n_symbol = correct_symbol = 0
    for r in cached:
        if r["symbol"]:
            n_symbol += 1
            ok = predict_symbol(r["letter"]) == r["answer"]
        else:
            n_alnum += 1
            best_w, best_p = None, float("-inf")
            for w, (la, lb, lg) in r["candidates"].items():
                p = w_a * (10 ** la) + w_b * (10 ** lb) + w_g * (10 ** lg)
                if p > best_p:
                    best_w, best_p = w, p
            ok = best_w == r["answer"]
        correct += ok
        if r["symbol"]:
            correct_symbol += ok
        else:
            correct_alnum += ok
    n = len(cached)
    return (correct / n, correct_alnum / n_alnum if n_alnum else 0.0,
            correct_symbol / n_symbol if n_symbol else 0.0)


def grid_search(cached, step):
    """Simplex grid over (w_a, w_b, w_g) summing to 1 -- coarse (step 0.1 -> 66 combos) is
    plenty since each point is cached-dict arithmetic, not a rescoring pass."""
    best = (None, -1.0)
    n_steps = round(1.0 / step)
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            w_a, w_b = i * step, j * step
            w_g = 1.0 - w_a - w_b
            if w_g < -1e-9:
                continue
            _, alnum_acc, _ = accuracy_at_weights(cached, w_a, w_b, w_g)
            if alnum_acc > best[1]:
                best = ((w_a, w_b, w_g), alnum_acc)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-csv", type=Path, default=ROOT / "data" / "devv_eval.csv")
    ap.add_argument("--test-csv", type=Path, default=ROOT / "data" / "devv_test.csv")
    ap.add_argument("--vocab", type=Path, default=ROOT / "weights" / "vocab.txt")
    ap.add_argument("--model-a", type=Path, default=RUN_DIR / "model_a.klm")
    ap.add_argument("--model-b", type=Path, default=RUN_DIR / "model_b.klm")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--cache", type=Path, default=ROOT / "weights" / "ensemble_cache.pkl",
                     help="scored-candidates cache, reused across runs -- delete to force a rescore")
    args = ap.parse_args()

    vocab_by_letter = load_vocab_by_letter(args.vocab)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.cache.exists():
        print(f"reusing cached scores -> {args.cache}")
        cached = pickle.loads(args.cache.read_bytes())
    else:
        model_a = kenlm.Model(str(args.model_a))
        model_b = kenlm.Model(str(args.model_b))
        gpt_model, gpt_vocab, id_to_tok, bucket_mask, seq_len = load_gpt(device)
        max_ctx = seq_len - 1
        cached = {
            "eval": cache_row_scores(args.eval_csv, args.limit, model_a, model_b, gpt_model, gpt_vocab,
                                      id_to_tok, bucket_mask, device, max_ctx, vocab_by_letter),
            "test": cache_row_scores(args.test_csv, args.limit, model_a, model_b, gpt_model, gpt_vocab,
                                      id_to_tok, bucket_mask, device, max_ctx, vocab_by_letter),
        }
        args.cache.write_bytes(pickle.dumps(cached))
        print(f"cached scores -> {args.cache}")

    (w_a, w_b, w_g), best_alnum = grid_search(cached["eval"], args.step)
    print(f"best weights (grid step {args.step}): model_a={w_a:.2f} model_b={w_b:.2f} gpt={w_g:.2f} "
          f"(eval alnum {best_alnum:.4f})")
    eval_acc, eval_alnum, eval_symbol = accuracy_at_weights(cached["eval"], w_a, w_b, w_g)
    test_acc, test_alnum, test_symbol = accuracy_at_weights(cached["test"], w_a, w_b, w_g)
    print(f"eval: overall {eval_acc:.4f}, alnum {eval_alnum:.4f}, symbol {eval_symbol:.4f}")
    print(f"test: overall {test_acc:.4f}, alnum {test_alnum:.4f}, symbol {test_symbol:.4f}")


if __name__ == "__main__":
    main()
