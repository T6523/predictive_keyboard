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

Perf history (500-row smoke tests, all on the same RTX 4060 laptop GPU under WSL2):
  1. naive: 0.41s/row -- looked like kenlm (BaseFullScore per candidate, full letter bucket,
     up to 10k+ candidates against model_b's 4.4GB order-9 trie) was the cost.
  2. shortlisted kenlm to GPT's top-200 (GPT's masked softmax already scores every bucket
     candidate for free, no second forward pass -- use it to pick which candidates are worth
     an expensive kenlm call): only 0.35s/row. Barely moved -- kenlm was never the bottleneck.
  3. profiled it for real (cProfile): score_candidates_gpt's per-candidate `.item()` call was
     a GPU sync PER CANDIDATE (thousands/row) -- fixed by one `.cpu().numpy()` sync for the
     whole vector, then plain-float indexing. Dropped to 0.13s/row, and confirmed dropping the
     shortlist back to exact (0) cost <10% -- kenlm was always cheap, this sync was the whole
     story. Shortlisting is gone entirely now (see #5) -- always exact.
  4. still 0.13s/row was 7x the ORIGINAL Kaggle GPT-only run's 0.0175s/row on the identical
     predict() call -- profiling pointed at the GPT forward pass itself, one row at a time
     (batch size 1). WSL2 adds real per-CUDA-call overhead; 94825 tiny unbatched forward
     passes pay that tax 94825 times. Fix: gpt_predict_batch() below batches --gpt-batch-size
     rows into one forward pass (left-padded, explicit position_ids so padding doesn't shift
     real tokens' positions -- HF's default position_ids assumes no padding, and logits_to_keep=1
     so lm_head only projects the LAST position, not the whole batch x seq_len x 99k-vocab
     tensor -- that OOM'd an 8GB GPU at batch=64 otherwise). Verified byte-close against
     infer.py's single-row predict() (see verify_batching()) before trusting it for real numbers.
  5. --limit 10000 (20k rows) stalled for 35+ min and drove RSS to 20.5GB, swapping out a
     29GB machine. Two compounding causes, both from storing exact per-candidate blends as a
     python dict-of-tuples per row: (a) ~34M (row, candidate) dict entries just for storage
     (~15GB for 20k rows at ~3400 avg candidates/row), and (b) grid_search's accuracy_at_weights
     looping every candidate in pure Python PER weight combo -- 66 combos x 34M entries = ~2.2
     BILLION scalar ops for eval alone. Fixed by switching the cache to one float32 (N,3) numpy
     probs array per row (aligned to vocab_by_letter[letter]'s order, ~20x less memory than the
     dict) with weighting done as a vectorized `probs @ weights` per row instead of a per-
     candidate python loop. This is also what made shortlisting (see #2) pointless to keep:
     the actual cost was never candidate count, it was doing the blend in the slowest possible
     way at that count.

Usage:
    python3 ensemble/ensemble_gpt_ngram.py                        # full eval+test grid search
    python3 ensemble/ensemble_gpt_ngram.py --limit 500 --step 0.2  # quick smoke test, coarse grid
    python3 ensemble/ensemble_gpt_ngram.py --verify-batching        # self-check, no scoring
"""
import argparse
import pickle
import sys
import time
import csv
from pathlib import Path

import kenlm
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gpt" / "33k_full_gpt"))
sys.path.insert(0, str(ROOT / "ngram"))

from predict_accuracy import load_vocab_by_letter, _prime  # noqa: E402 -- reuse, don't duplicate
from infer import load_model as load_gpt  # noqa: E402

RUN_DIR = ROOT / "weights" / "run_20260905_103654"
LN10 = 2.302585092994046


def score_candidates_kenlm(model, context_tokens, candidates):
    """log10 P(word | context) for every candidate, from a context state primed once, as a
    float32 array aligned to candidates' order."""
    s1, s2 = kenlm.State(), kenlm.State()
    state = _prime(model, context_tokens, s1, s2)
    out = kenlm.State()
    return np.array([model.BaseFullScore(state, w, out).log_prob for w in candidates], dtype=np.float32)


@torch.no_grad()
def gpt_predict_batch(model, gpt_vocab, device, batch_tokens, max_ctx):
    """Batched replacement for infer.py's predict() -- one forward pass for the whole batch
    instead of one per row. Left-pads with <s> so every sequence's real last token lands at
    position -1 (matches predict()'s `logits[0, -1]` contract), with explicit position_ids
    (attention_mask cumsum) so left-padding doesn't shift real tokens off their true positions
    -- HF's default position_ids = arange(seq_len) assumes no padding and would silently
    mis-position every row shorter than the batch max. Returns one full-vocab logit row per
    input, unmasked (same as predict()'s second return value)."""
    pad_id = gpt_vocab["<s>"]
    id_seqs = [[pad_id] + [gpt_vocab.get(t, pad_id) for t in toks][-max_ctx:] for toks in batch_tokens]
    max_len = max(len(s) for s in id_seqs)
    input_ids = torch.full((len(id_seqs), max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((len(id_seqs), max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(id_seqs):
        input_ids[i, max_len - len(seq):] = torch.tensor(seq, device=device)
        attn_mask[i, max_len - len(seq):] = 1
    position_ids = (attn_mask.cumsum(-1) - 1).clamp(min=0)
    # logits_to_keep=1: lm_head only projects the LAST position through the 99k vocab, not
    # every position in the batch -- omitting this OOM'd at batch=64 (batch x seq_len x vocab
    # logits materialized for the whole sequence, same class of bug qwen's eval_qwen.py already
    # named and fixed for its own model, see that file's docstring).
    return model(input_ids, attention_mask=attn_mask, position_ids=position_ids,
                 logits_to_keep=1).logits[:, -1]


def verify_batching(gpt_model, gpt_vocab, device, max_ctx, eval_csv, n=16):
    """ponytail: batching's whole failure mode is silent position/padding bugs (right answer
    shape, wrong numbers) -- this is the one check that would catch it. Compares
    gpt_predict_batch's output against infer.py's original single-row predict() on real rows
    of mixed lengths (padding only bites when lengths differ within a batch)."""
    with open(eval_csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["context"].split()][:n]
    batch_tokens = [r["context"].split() for r in rows]
    batched = gpt_predict_batch(gpt_model, gpt_vocab, device, batch_tokens, max_ctx)
    # infer.py's predict() needs id_to_tok/bucket_mask only for its masked-argmax branch, which
    # this script never uses -- call the model directly instead, same as predict() does internally.
    max_err = 0.0
    for i, toks in enumerate(batch_tokens):
        ids = [gpt_vocab["<s>"]] + [gpt_vocab.get(t, gpt_vocab["<s>"]) for t in toks][-max_ctx:]
        x = torch.tensor([ids], device=device)
        single_logits = gpt_model(x).logits[0, -1]
        err = (batched[i] - single_logits).abs().max().item()
        max_err = max(max_err, err)
    status = "PASS" if max_err < 1e-3 else "FAIL"
    print(f"verify_batching: {n} rows, max abs logit diff {max_err:.6f} -- {status}")
    assert status == "PASS", "batched GPT forward doesn't match single-row -- padding/position bug"


def score_candidates_gpt(logits, gpt_vocab, candidates):
    """log10 P(word | context) for every candidate, sliced out of the one masked logit vector
    infer.py's predict() already computed for this row -- no second GPT forward pass. Returns
    a float32 array aligned to candidates' order (-30.0 floor for words outside GPT's vocab).

    One .cpu() sync for the whole vector, then plain indexing -- calling .item() per candidate
    (the original code) forces a GPU sync per call, which for a 3000+-candidate letter bucket
    dwarfed even the unshortlisted kenlm cost (measured: shortlisting kenlm to top-200 barely
    moved wall time, because this loop was the real bottleneck all along)."""
    logprobs = F.log_softmax(logits.float(), dim=-1).cpu().numpy()
    out = np.full(len(candidates), -30.0, dtype=np.float32)
    for i, w in enumerate(candidates):
        wid = gpt_vocab.get(w)
        if wid is not None:
            out[i] = logprobs[wid] / LN10
    return out


def cache_row_scores(path, limit, model_a, model_b, gpt_model, gpt_vocab, device, max_ctx,
                      vocab_by_letter, batch_size):
    from scripts.symbol_predict import is_symbol_letter

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]

    # Pass 1: GPT is the expensive part (one forward pass/row) -- batch every alnum row through
    # it batch_size at a time instead of one at a time (see module docstring, perf history #4).
    # Symbol rows never touch the model.
    alnum_idx = [i for i, row in enumerate(rows) if not is_symbol_letter(row["first letter"])]
    logits_by_idx = {}
    t0 = time.time()
    for start in range(0, len(alnum_idx), batch_size):
        chunk = alnum_idx[start:start + batch_size]
        batch_tokens = [rows[i]["context"].split() for i in chunk]
        logits = gpt_predict_batch(gpt_model, gpt_vocab, device, batch_tokens, max_ctx)
        for j, i in enumerate(chunk):
            logits_by_idx[i] = logits[j]
        if (start // batch_size) % 50 == 0:
            print(f"  GPT-scored {min(start + batch_size, len(alnum_idx))}/{len(alnum_idx)} "
                  f"({time.time()-t0:.0f}s)")
    print(f"{path.name}: GPT-scored {len(alnum_idx)} rows ({time.time()-t0:.0f}s)")

    # Pass 2: kenlm scoring, cheap per-row (confirmed: never the bottleneck, see docstring #2-3).
    # Stores one float32 (N,3) probs array per row -- NOT a dict-of-tuples (see docstring #5,
    # that was ~20x more memory and made grid_search do a per-candidate python loop per weight
    # combo). candidates itself is the SAME list object as vocab_by_letter[letter] every time
    # (never copied), so this only costs one small array per row.
    cached = []
    t0 = time.time()
    for n, row in enumerate(rows):
        letter, answer = row["first letter"], row["answer"]
        if is_symbol_letter(letter):
            cached.append({"letter": letter, "answer": answer, "symbol": True})
        else:
            tokens = row["context"].split()
            candidates = vocab_by_letter.get(letter.lower(), ())
            if not candidates:
                cached.append({"letter": letter, "answer": answer, "symbol": False,
                                "candidates": candidates, "probs": np.zeros((0, 3), dtype=np.float32),
                                "answer_idx": -1})
                continue
            log_g = score_candidates_gpt(logits_by_idx[n], gpt_vocab, candidates)
            log_a = score_candidates_kenlm(model_a, tokens, candidates)
            log_b = score_candidates_kenlm(model_b, tokens, candidates)
            probs = np.power(10.0, np.stack([log_a, log_b, log_g], axis=1))
            answer_idx = candidates.index(answer) if answer in candidates else -1
            cached.append({"letter": letter, "answer": answer, "symbol": False,
                            "candidates": candidates, "probs": probs, "answer_idx": answer_idx})
        if (n + 1) % 5000 == 0:
            print(f"  kenlm-scored {n+1}/{len(rows)} ({time.time()-t0:.0f}s)")
    print(f"{path.name}: kenlm-scored {len(rows)} rows ({time.time()-t0:.0f}s)")
    return cached


def accuracy_at_weights(cached, w_a, w_b, w_g):
    """Vectorized: probs @ weights + argmax per row, not a per-candidate python loop (see
    docstring #5 -- that was the real bottleneck at scale, not scoring)."""
    from scripts.symbol_predict import predict_symbol

    weights = np.array([w_a, w_b, w_g], dtype=np.float64)
    correct = n_alnum = correct_alnum = n_symbol = correct_symbol = 0
    for r in cached:
        if r["symbol"]:
            n_symbol += 1
            ok = predict_symbol(r["letter"]) == r["answer"]
        else:
            n_alnum += 1
            ok = r["probs"].size > 0 and int(np.argmax(r["probs"] @ weights)) == r["answer_idx"]
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
    plenty since each point is a vectorized pass over cached probs, not a rescoring pass."""
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
    ap.add_argument("--gpt-batch-size", type=int, default=64,
                     help="rows per GPT forward pass -- the actual bottleneck fix, see module docstring")
    ap.add_argument("--verify-batching", action="store_true",
                     help="self-check: batched GPT output vs single-row predict(), then exit")
    ap.add_argument("--cache", type=Path, default=ROOT / "weights" / "ensemble_cache.pkl",
                     help="scored-candidates cache, reused across runs -- delete to force a rescore")
    args = ap.parse_args()

    vocab_by_letter = load_vocab_by_letter(args.vocab)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.verify_batching:
        gpt_model, gpt_vocab, _id_to_tok, _bucket_mask, seq_len = load_gpt(device)
        verify_batching(gpt_model, gpt_vocab, device, seq_len - 1, args.eval_csv)
        return

    if args.cache.exists():
        print(f"reusing cached scores -> {args.cache}")
        cached = pickle.loads(args.cache.read_bytes())
    else:
        model_a = kenlm.Model(str(args.model_a))
        model_b = kenlm.Model(str(args.model_b))
        gpt_model, gpt_vocab, _id_to_tok, _bucket_mask, seq_len = load_gpt(device)
        max_ctx = seq_len - 1
        cached = {
            "eval": cache_row_scores(args.eval_csv, args.limit, model_a, model_b, gpt_model, gpt_vocab,
                                      device, max_ctx, vocab_by_letter, args.gpt_batch_size),
            "test": cache_row_scores(args.test_csv, args.limit, model_a, model_b, gpt_model, gpt_vocab,
                                      device, max_ctx, vocab_by_letter, args.gpt_batch_size),
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
