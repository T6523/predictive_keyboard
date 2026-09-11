"""Local version of kaggle/matched_eval's notebook -- Kaggle's T4 turned out no
faster here (1.5 rows/s even after the device-sharding fix, vs local's 7.6), so run
the matched zero-shot-vs-fine-tuned comparison on the 4060 instead. Same protocol
both times (beam k=5), same stratified 10% word rows -- resolves whether the LoRA
fine-tune actually helped (the old comparison mixed greedy zero-shot vs beam-top1
fine-tuned, not apples to apples). Also exports per-candidate log-prob scores for
blend_ngram.py's n-gram blend.

Usage:
    python3 gen_matched_scores.py --lora ../weights/qwen3b_lora_kaggle \
        --out ../weights/qwen_matched_scores.csv
"""
import argparse
import csv
import random
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from infer_topk import (
    build_letter_masks,
    categorize,
    detect_boundary,
    get_boundary_ids,
    predict_topk_batch,
)

MODEL_ID = "Qwen/Qwen2.5-3B"


def run_pass(model, tokenizer, word_rows, masks, boundary_ids_tensor, device, k, batch_size, label):
    model.generation_config.max_length = None
    preds = {}
    t0 = time.time()
    for start in range(0, len(word_rows), batch_size):
        batch = word_rows[start : start + batch_size]
        contexts = [r["context"] for r in batch]
        letters = [r["first letter"] for r in batch]
        ranked = predict_topk_batch(model, tokenizer, contexts, letters, masks, boundary_ids_tensor, device, k, with_scores=True)
        for r, cands in zip(batch, ranked):
            preds[id(r)] = cands
        done = start + len(batch)
        if done % (batch_size * 10) == 0 or done == len(word_rows):
            elapsed = time.time() - t0
            print(f"[{label}] [{done}/{len(word_rows)}] {elapsed:.1f}s, {done/max(elapsed,1e-9):.2f} rows/s", flush=True)
    return preds


def fmt(cands):
    return "; ".join(f"{w}:{sc:.4f}" for w, sc in cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--lora", default="../weights/qwen3b_lora_kaggle")
    ap.add_argument("--dev", default="../data/dev_set_final.csv")
    ap.add_argument("--out", default="../weights/qwen_matched_scores.csv")
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    boundary = detect_boundary(tokenizer)

    with open(args.dev, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # same stratified 10% sample as every other eval this track -- seed 42
    random.seed(args.seed)
    by_cat = {}
    for r in all_rows:
        by_cat.setdefault(categorize(r["answer"]), []).append(r)
    sample = []
    for cat, group in by_cat.items():
        n = max(1, round(len(group) * args.frac))
        sample.extend(random.sample(group, n))
    word_rows = [r for r in sample if categorize(r["answer"]) == "word"]
    other_rows = [r for r in sample if categorize(r["answer"]) != "word"]
    print(f"{len(sample)} stratified rows -- {len(word_rows)} word rows through both models, {len(other_rows)} passed through untouched")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto")
    model.eval()

    vocab_size = model.get_output_embeddings().weight.shape[0]
    masks = build_letter_masks(tokenizer, boundary, device, vocab_size)
    boundary_ids_tensor = torch.tensor(get_boundary_ids(tokenizer, boundary), device=device)

    zeroshot_preds = run_pass(model, tokenizer, word_rows, masks, boundary_ids_tensor, device, args.k, args.batch_size, "zero-shot")

    model = PeftModel.from_pretrained(model, args.lora)  # same quantized base, adapter on top
    model.eval()
    lora_preds = run_pass(model, tokenizer, word_rows, masks, boundary_ids_tensor, device, args.k, args.batch_size, "lora")

    def acc(preds, label):
        c1 = c5 = 0
        for r in word_rows:
            ans = r["answer"].strip().lower()
            cands = [w.lower() for w, _ in preds[id(r)]]
            c1 += cands[:1] == [ans]
            c5 += ans in cands[:5]
        n = len(word_rows)
        print(f"{label:10s} top1 {c1}/{n} = {c1/n*100:.2f}%  top5 {c5}/{n} = {c5/n*100:.2f}%")

    print(f"\n--- matched comparison, same {len(word_rows)} word rows, beam k={args.k} ---")
    acc(zeroshot_preds, "zero-shot")
    acc(lora_preds, "lora")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["context", "first_letter", "answer", "category", "zeroshot_preds", "lora_preds"])
        for r in word_rows:
            w.writerow([r["context"], r["first letter"], r["answer"], "word",
                        fmt(zeroshot_preds[id(r)]), fmt(lora_preds[id(r)])])
        for r in other_rows:
            w.writerow([r["context"], r["first letter"], r["answer"], categorize(r["answer"]), "", ""])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
