"""Score LM top5 (union) n-gram top10 candidates directly with one teacher-forced
batched forward pass per row (status.md advice item 5), instead of trusting
generate()'s beam sequence_scores. A single forward pass gives every candidate's
FULL-word log-prob under teacher forcing -- no beam-search pruning on multi-subword
words, and scores across differently-shaped candidates are directly comparable
(beam search can silently drop a candidate whose partial prefix scored low early,
even if its full continuation would have been fine). Also gives item 2's learned
blender a clean, consistent LM feature to build on.

Usage:
    python3 score_candidates.py --lora ../weights/qwen3b_lora_kaggle \
        --data ../weights/qwen_matched_scores.csv --ngram ../weights/ngram_4_a.bin \
        --out ../weights/qwen_tf_scores.csv
"""
import argparse
import csv
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from blend_ngram import parse_preds
from infer_ngram import load_model, topk_by_letter

MODEL_ID = "Qwen/Qwen2.5-3B"


@torch.inference_mode()
def score_candidates_batch(model, tokenizer, context, candidates, device):
    """Teacher-forced full-word log-prob (natural log, summed over subword tokens)
    for each candidate -- one forward pass covers the whole row's candidate set."""
    ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    prompt_len = len(ctx_ids)
    seqs, cand_lens = [], []
    for cand in candidates:
        full_ids = tokenizer(context + " " + cand, add_special_tokens=False)["input_ids"]
        seqs.append(full_ids)
        cand_lens.append(len(full_ids) - prompt_len)

    max_len = max(len(s) for s in seqs)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)

    logits = model(input_ids=input_ids, attention_mask=attn).logits
    logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = input_ids[:, 1:]
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, max_len-1)

    start = prompt_len - 1  # token_lp[:, start] predicts input_ids[:, prompt_len], the first candidate token
    return [token_lp[i, start : start + clen].sum().item() for i, clen in enumerate(cand_lens)]


def fmt(cands_scores):
    return "; ".join(f"{w}:{sc:.4f}" for w, sc in cands_scores)


def demo():
    """Self-check: context+candidate tokenization stays prefix-stable (the slicing
    in score_candidates_batch assumes tokenizing context+" "+cand reproduces the
    context's own tokens as an exact prefix -- BPE merges across the boundary would
    break this silently). No model forward needed."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ctx = "the cat sat on the"
    ctx_ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
    for cand in ["mat", "roof", "a", "an", "x"]:
        full_ids = tokenizer(ctx + " " + cand, add_special_tokens=False)["input_ids"]
        assert full_ids[: len(ctx_ids)] == ctx_ids, f"prefix mismatch for {cand!r}"

    # --lm-source topk's plain-word-list parsing (infer_topk.py's top_predictions
    # column has no ":score" suffix, unlike zeroshot/lora/tf's parse_preds format)
    field = "the; a; an"
    assert [w.strip().lower() for w in field.split("; ") if w.strip()] == ["the", "a", "an"]

    print("demo ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--lora", default="../weights/qwen3b_lora_kaggle")
    ap.add_argument("--data", default="../weights/qwen_matched_scores.csv")
    ap.add_argument("--ngram", default="../weights/ngram_4_a.bin")
    ap.add_argument("--ngram-topk", type=int, default=10)
    ap.add_argument("--out", default="../weights/qwen_tf_scores.csv")
    ap.add_argument("--lm-source", choices=["zeroshot", "lora", "tf", "topk"], default="lora",
                     help="'tf' reads qwen_tf_scores.csv's own tf_preds column -- reuses that exact "
                          "candidate set (LM top5 union ngram top10) to score with a different model, "
                          "e.g. Mistral-7B for the item-4 blend, so the two models are compared on "
                          "identical candidates. 'topk' reads infer_topk.py's own top_predictions "
                          "column (plain '; '-joined words, no ':score' suffix -- beam-k10's own "
                          "output format, not the earlier zeroshot/lora single-generation format)")
    ap.add_argument("--limit", type=int, default=None, help="only score the first N word rows (timing runs)")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.lora or args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map={"": 0})
    if args.lora:
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    n, counts, vocab, id_to_tok = load_model(args.ngram)

    with open(args.data, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    word_rows = [r for r in rows if r["category"] == "word"]
    other_rows = [r for r in rows if r["category"] != "word"]
    if args.limit:
        word_rows = word_rows[: args.limit]
    pred_col = "top_predictions" if args.lm_source == "topk" else f"{args.lm_source}_preds"

    t0 = time.time()
    out_rows = []
    for i, r in enumerate(word_rows):
        if args.lm_source == "topk":
            beam_cands = [w.strip().lower() for w in r[pred_col].split("; ") if w.strip()]
        else:
            beam_cands = [w for w, _ in parse_preds(r[pred_col])]
        ngram_cands = topk_by_letter(r["context"].split(), r["first_letter"], n, counts, vocab, id_to_tok, args.ngram_topk)
        candidates = list(dict.fromkeys(beam_cands + ngram_cands))  # union, order-preserving
        scores = score_candidates_batch(model, tokenizer, r["context"], candidates, device)
        out_rows.append((r, list(zip(candidates, scores))))
        if (i + 1) % 200 == 0 or i + 1 == len(word_rows):
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(word_rows)}] {elapsed:.1f}s, {(i+1)/elapsed:.2f} rows/s", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["context", "first_letter", "answer", "category", "tf_preds"])
        for r, cands_scores in out_rows:
            cands_scores.sort(key=lambda x: -x[1])
            w.writerow([r["context"], r["first_letter"], r["answer"], "word", fmt(cands_scores)])
        if not args.limit:
            for r in other_rows:
                w.writerow([r["context"], r["first_letter"], r["answer"], r["category"], ""])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
