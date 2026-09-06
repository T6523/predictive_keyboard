#!/usr/bin/env python3
"""Inference for the from-scratch GPT2 checkpoint in this folder (gpt2.pt -- torch.save output,
just torch's own zip container, nothing to unzip). Loads vocab/config straight from the
checkpoint (mirrors gpt/kernel/build_notebook.py's predict()/run_eval(), same masked-
argmax contract), scores devv_eval.csv / devv_test.csv, and writes a predictions csv per file
into this same folder so the result can be compared against the n-gram runs.

Output schema deliberately matches the two things weights/run_*/'s n-gram csvs already give you:
  - predicted/is_correct: same next-word-argmax task ngram/predict_accuracy.py scores for the n-gram
    (that script prints accuracy but never saves a csv -- this one does, for both models).
  - gpt_logprob10: log10 P(true answer | context) under this model, same idea as
    model_a_logprob10/model_b_logprob10 in weights/run_*/devv_eval_predictions.csv -- lets you
    reuse ngram/interpolate.py's grid-search machinery to blend GPT with the n-grams later.
Symbol-letter rows (routed via scripts/symbol_predict.py, same convention as everywhere else in
this repo) get predicted/is_correct but no gpt_logprob10 (model never queried, nothing to score).

Needs torch + transformers (models.py builds the checkpoint's GPT2LMHeadModel via HF's
GPT2Config/from_config) -- not in this repo's local .venv by default; pip install both first
if running here instead of on Kaggle.

Usage:
    python3 gpt/33k_full_gpt/infer.py                      # full devv_eval.csv + devv_test.csv
    python3 gpt/33k_full_gpt/infer.py --limit 500          # smoke test
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
TRANSFORMER_DIR = HERE.parent
ROOT = TRANSFORMER_DIR.parent
sys.path.insert(0, str(TRANSFORMER_DIR))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import build_bucket_table, BUCKETS  # noqa: E402
from models import build_model  # noqa: E402
from symbol_predict import is_symbol_letter, predict_symbol  # noqa: E402

CKPT_PATH = HERE / "gpt2.pt"


def load_model(device):
    ckpt = torch.load(CKPT_PATH, map_location=device)
    vocab, id_to_tok, config = ckpt["vocab"], ckpt["id_to_tok"], ckpt["config"]
    model = build_model(config["model_name"], config["vocab_size"], n_layer=config["n_layer"],
                         n_head=config["n_head"], n_embd=config["n_embd"], seq_len=config["seq_len"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"loaded {config['model_name']} @ step {ckpt.get('step', '?')}, vocab {len(vocab)} tokens")

    bucket_table = build_bucket_table(id_to_tok)
    bucket_bool_mask = torch.zeros(len(BUCKETS), len(vocab), dtype=torch.bool, device=device)
    for i, b in enumerate(bucket_table):
        if b >= 0:
            bucket_bool_mask[b, i] = True
    return model, vocab, id_to_tok, bucket_bool_mask, config["seq_len"]


@torch.no_grad()
def predict(model, vocab, id_to_tok, bucket_bool_mask, device, context_tokens, letter, max_ctx):
    """Same contract as build_notebook.py's predict(): last-position logits, masked to the
    letter's bucket, argmax. Also returns log10 P(answer) computed separately by the caller
    (needs the unmasked distribution, so predict() hands back raw logits too)."""
    ids = [vocab["<s>"]] + [vocab.get(t, vocab["<s>"]) for t in context_tokens][-max_ctx:]
    x = torch.tensor([ids], device=device)
    logits = model(x).logits[0, -1]
    b = BUCKETS.index(letter.lower())
    masked = logits.masked_fill(~bucket_bool_mask[b], float("-inf"))
    best = masked.argmax().item()
    pred = id_to_tok[best] if masked[best] != float("-inf") else None
    return pred, logits


def run_eval(path, model, vocab, id_to_tok, bucket_bool_mask, device, max_ctx, limit, out_csv):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]

    correct = n_symbol = correct_symbol = n_alnum = correct_alnum = 0
    out_rows = []
    t0 = time.time()
    for n, row in enumerate(rows, 1):
        letter, answer = row["first letter"], row["answer"]
        gpt_logprob10 = None
        if is_symbol_letter(letter):
            n_symbol += 1
            pred = predict_symbol(letter)
        else:
            n_alnum += 1
            tokens = row["context"].split()
            pred, logits = predict(model, vocab, id_to_tok, bucket_bool_mask, device, tokens, letter, max_ctx)
            ans_id = vocab.get(answer)
            if ans_id is not None:
                logprob_e = F.log_softmax(logits.float(), dim=-1)[ans_id].item()
                gpt_logprob10 = logprob_e / 2.302585092994046  # ln -> log10, matches kenlm's convention
        is_correct = pred == answer
        correct += is_correct
        if is_symbol_letter(letter):
            correct_symbol += is_correct
        else:
            correct_alnum += is_correct
        out_rows.append({"context": row["context"], "first letter": letter, "answer": answer,
                          "predicted": pred, "is_correct": is_correct, "gpt_logprob10": gpt_logprob10})
        if n % 2000 == 0:
            print(f"  {n}/{len(rows)} ({time.time() - t0:.0f}s)")

    n = len(rows)
    print(f"{path.name}: rows {n}, eval time {time.time() - t0:.1f}s")
    print(f"  accuracy: {correct}/{n} = {correct / max(n, 1):.4f}")
    if n_alnum:
        print(f"  alnum route:  {correct_alnum}/{n_alnum} = {correct_alnum / n_alnum:.4f}")
    if n_symbol:
        print(f"  symbol route: {correct_symbol}/{n_symbol} = {correct_symbol / n_symbol:.4f}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["context", "first letter", "answer", "predicted",
                                          "is_correct", "gpt_logprob10"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"  -> {out_csv}")
    return correct / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-csv", type=Path, default=ROOT / "data" / "devv_eval.csv")
    ap.add_argument("--test-csv", type=Path, default=ROOT / "data" / "devv_test.csv")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per file (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    model, vocab, id_to_tok, bucket_bool_mask, seq_len = load_model(device)
    max_ctx = seq_len - 1

    run_eval(args.eval_csv, model, vocab, id_to_tok, bucket_bool_mask, device, max_ctx,
              args.limit, HERE / "devv_eval_predictions.csv")
    run_eval(args.test_csv, model, vocab, id_to_tok, bucket_bool_mask, device, max_ctx,
              args.limit, HERE / "devv_test_predictions.csv")


if __name__ == "__main__":
    main()
