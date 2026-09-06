#!/usr/bin/env python3
"""Standalone Kaggle script-kernel version of ../33k_full_gpt/infer.py -- same masked-
argmax contract, same output schema (context,first letter,answer,predicted,is_correct,
gpt_logprob10), but self-contained (no repo imports -- a script kernel only gets this one file)
and resolving paths under /kaggle/input instead of the local repo tree.

Attach two datasets: teekn07/predictive-keyboard-ckpt (gpt2.pt) and teekn07/keyboard
(devv_eval.csv, devv_test.csv) -- both already exist, no new upload needed.

Outputs -> /kaggle/working (the only writable dir in a script kernel; /kaggle/src is read-only,
same gotcha as the Qwen kernel's earlier crash).
"""
import csv
import glob
import string
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

WORK_DIR = Path("/kaggle/working")
BUCKETS = list(string.ascii_lowercase) + list(string.digits)
LN10 = 2.302585092994046


# --- inlined from scripts/symbol_predict.py (standalone kernel, can't import across the repo) ---
def predict_symbol(letter):
    return "[UNK]" if letter == "[" else letter


def is_symbol_letter(letter):
    return not str(letter).isalnum()


# --- inlined from ../vocab.py (just the bit infer needs: id -> bucket) ---
def build_bucket_table(id_to_tok):
    table = []
    for tok in id_to_tok:
        if not tok or not tok[0].isalnum():
            table.append(-1)
        else:
            ch = tok[0].lower()
            table.append(BUCKETS.index(ch) if ch in BUCKETS else -1)
    return table


def resolve(filename):
    hits = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{filename} not found under /kaggle/input")
    return hits[0]


def load_model(device):
    ckpt = torch.load(resolve("gpt2.pt"), map_location=device)
    vocab, id_to_tok, config = ckpt["vocab"], ckpt["id_to_tok"], ckpt["config"]
    cfg = GPT2Config(vocab_size=config["vocab_size"], n_positions=config["seq_len"],
                      n_embd=config["n_embd"], n_layer=config["n_layer"], n_head=config["n_head"])
    model = GPT2LMHeadModel(cfg)
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
    ids = [vocab["<s>"]] + [vocab.get(t, vocab["<s>"]) for t in context_tokens][-max_ctx:]
    x = torch.tensor([ids], device=device)
    logits = model(x).logits[0, -1]
    b = BUCKETS.index(letter.lower())
    masked = logits.masked_fill(~bucket_bool_mask[b], float("-inf"))
    best = masked.argmax().item()
    pred = id_to_tok[best] if masked[best] != float("-inf") else None
    return pred, logits


def run_eval(path, model, vocab, id_to_tok, bucket_bool_mask, device, max_ctx, out_csv):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

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
                gpt_logprob10 = logprob_e / LN10
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
    print(f"{Path(path).name}: rows {n}, eval time {time.time() - t0:.1f}s")
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, id_to_tok, bucket_bool_mask, seq_len = load_model(device)
    max_ctx = seq_len - 1

    run_eval(resolve("devv_eval.csv"), model, vocab, id_to_tok, bucket_bool_mask, device, max_ctx,
              WORK_DIR / "devv_eval_predictions.csv")
    run_eval(resolve("devv_test.csv"), model, vocab, id_to_tok, bucket_bool_mask, device, max_ctx,
              WORK_DIR / "devv_test_predictions.csv")


if __name__ == "__main__":
    main()
