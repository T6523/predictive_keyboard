"""Top-1/5/10 accuracy via beam search, on a stratified sample of dev_set_final.csv.
Mirrors infer_transformer.py's letter-mask + boundary-stop machinery, but beam search
needs the letter mask expanded per-beam and returns num_beams sequences per input row
instead of one -- different enough plumbing to duplicate rather than import.

Top-k here means top-k DISTINCT predicted words (beam duplicates collapsed, rank order
kept) -- matches what a real keyboard suggestion bar would show, not raw beam count.

Usage:
    python3 infer_topk.py --lora ../weights/qwen3b_lora_kaggle --k 10 \
        --frac 0.10 --seed 42 --out ../weights/dev_predictions_topk.csv
"""
import argparse
import csv
import random
import re
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    StoppingCriteria,
)

MODEL_ID = "Qwen/Qwen2.5-3B"

CAT_NUMBER = re.compile(r"[0-9]+")
CAT_WORD = re.compile(r"(?=.*[a-z])[a-z']+", re.IGNORECASE)  # >=1 real letter -- a bare "'" isn't a word


def categorize(tok):
    if CAT_NUMBER.fullmatch(tok):
        return "number"
    if CAT_WORD.fullmatch(tok):
        return "word"
    return "symbol"


def detect_boundary(tokenizer):
    vocab = tokenizer.get_vocab()
    counts = {"▁": 0, "Ġ": 0}
    for piece in vocab:
        if piece[:1] in counts:
            counts[piece[:1]] += 1
    return max(counts, key=counts.get)


def build_letter_masks(tokenizer, boundary, device, vocab_size):
    vocab = tokenizer.get_vocab()
    letters = list("abcdefghijklmnopqrstuvwxyz0123456789")
    masks = {c: torch.zeros(vocab_size, dtype=torch.bool) for c in letters}
    for piece, idx in vocab.items():
        if len(piece) > 1 and piece[0] == boundary and piece[1].lower() in masks:
            masks[piece[1].lower()][idx] = True
    return {c: m.to(device) for c, m in masks.items()}


def get_boundary_ids(tokenizer, boundary):
    return [idx for piece, idx in tokenizer.get_vocab().items() if piece.startswith(boundary)]


class FirstTokenLetterMask(LogitsProcessor):
    """Same as infer_transformer.py's, but letter_mask must already be expanded to
    (batch * num_beams, vocab) -- generate() replicates input_ids by num_beams before
    the first forward pass, so scores arrive in that same expanded shape at every step,
    including step 0."""

    def __init__(self, letter_mask, prompt_len):
        self.letter_mask = letter_mask
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] == self.prompt_len:
            scores = scores.masked_fill(~self.letter_mask, float("-inf"))
        return scores


class StopAtWordBoundary(StoppingCriteria):
    def __init__(self, prompt_len, boundary_ids_tensor, eos_id):
        self.prompt_len = prompt_len
        self.boundary_ids_tensor = boundary_ids_tensor
        self.eos_id = eos_id

    def __call__(self, input_ids, scores, **kwargs):
        cur_len = input_ids.shape[1]
        if cur_len <= self.prompt_len + 1:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        last = input_ids[:, -1]
        return torch.isin(last, self.boundary_ids_tensor) | (last == self.eos_id)


@torch.inference_mode()
def predict_topk_batch(model, tokenizer, contexts, letters, masks, boundary_ids_tensor, device, k, max_extra=4, with_scores=False):
    """Beam search, num_beams=num_return_sequences=k. Returns one list of up to k
    distinct predicted words per input row, best-first (or (word, score) pairs if
    with_scores -- score is generate()'s own length-normalized sequence log-prob,
    needed for blending with the n-gram, see blend_ngram.py)."""
    enc = tokenizer(contexts, return_tensors="pt", padding=True).to(device)
    letter_mask = torch.stack([masks[l.lower()] for l in letters])  # (B, vocab)
    letter_mask = letter_mask.repeat_interleave(k, dim=0)  # (B*k, vocab) -- matches beam expansion order
    prompt_len = enc["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id

    out = model.generate(
        **enc,
        max_new_tokens=max_extra + 1,
        num_beams=k,
        num_return_sequences=k,
        do_sample=False,
        early_stopping=True,
        logits_processor=[FirstTokenLetterMask(letter_mask, prompt_len)],
        stopping_criteria=[StopAtWordBoundary(prompt_len, boundary_ids_tensor, eos_id)],
        pad_token_id=eos_id,
        return_dict_in_generate=with_scores,
        output_scores=with_scores,  # return_dict_in_generate alone leaves sequences_scores None
    )
    sequences = out.sequences if with_scores else out
    seq_scores = out.sequences_scores.tolist() if with_scores else None
    generated = sequences[:, prompt_len:].tolist()  # (B*k, ...), ordered per input row then per beam rank

    boundary_set = set(boundary_ids_tensor.tolist())
    results = []
    for i in range(0, len(generated), k):
        seen, ranked = set(), []
        row_scores = seq_scores[i : i + k] if with_scores else [None] * k
        for row, sc in zip(generated[i : i + k], row_scores):
            cut = next((j for j, tid in enumerate(row) if j > 0 and (tid in boundary_set or tid == eos_id)), len(row))
            word = tokenizer.decode(row[:cut]).strip()
            key = word.lower()
            if key not in seen:
                seen.add(key)
                ranked.append((word, sc) if with_scores else word)
        results.append(ranked)
    return results


def demo():
    """Self-check: categorize() + dedupe-preserving-rank logic, no model/network needed."""
    assert categorize("dog") == "word"
    assert categorize("'") == "symbol"
    assert categorize("don't") == "word"
    assert categorize("123") == "number"
    assert categorize(",") == "symbol"

    def dedupe(words):
        seen, out = set(), []
        for w in words:
            if w.lower() not in seen:
                seen.add(w.lower())
                out.append(w)
        return out

    assert dedupe(["The", "the", "That", "the"]) == ["The", "That"]
    print("demo ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--dev", default="../data/dev_set_final.csv")
    ap.add_argument("--out", default="../weights/dev_predictions_topk.csv")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--frac", type=float, default=0.10, help="stratified-by-category sample fraction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=10, help="top-k, also num_beams")
    ap.add_argument("--batch-size", type=int, default=4, help="beam search multiplies VRAM ~k x -- keep small")
    ap.add_argument("--max-extra", type=int, default=4)
    ap.add_argument("--print-sample", type=int, default=10)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.lora or args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    boundary = detect_boundary(tokenizer)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto")
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()
    model.generation_config.max_length = None

    vocab_size = model.get_output_embeddings().weight.shape[0]
    masks = build_letter_masks(tokenizer, boundary, device, vocab_size)
    boundary_ids_tensor = torch.tensor(get_boundary_ids(tokenizer, boundary), device=device)

    with open(args.dev, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Stratified sample by category, same convention as the Kaggle notebook's 10% run.
    random.seed(args.seed)
    by_cat = {}
    for r in rows:
        by_cat.setdefault(categorize(r["answer"]), []).append(r)
    rows = []
    for cat, group in by_cat.items():
        n = max(1, round(len(group) * args.frac))
        rows.extend(random.sample(group, n))
    print(f"stratified {args.frac*100:.0f}% sample: {len(rows)} rows "
          f"(source sizes: {{c: len(g) for c, g in by_cat.items()}})"
          .replace("{c: len(g) for c, g in by_cat.items()}", str({c: len(g) for c, g in by_cat.items()})))

    # Symbol rows: deterministic single-candidate prediction (letter copy). "top-k"
    # doesn't add anything real here -- record it once, applied to all three top-N.
    alpha_rows = [r for r in rows if r["first letter"].isalpha()]
    digit_rows = [r for r in rows if r["first letter"].isdigit()]  # number: also goes through
    symbol_rows = [r for r in rows if not r["first letter"].isalnum()]  # the model, masked to length

    results = []  # (row, ranked_predictions)
    for r in symbol_rows:
        results.append((r, [r["first letter"]]))

    alpha_rows = alpha_rows + digit_rows

    t0 = time.time()
    printed = 0
    for start in range(0, len(alpha_rows), args.batch_size):
        batch = alpha_rows[start : start + args.batch_size]
        contexts = [r["context"] for r in batch]
        letters = [r["first letter"] for r in batch]
        preds = predict_topk_batch(model, tokenizer, contexts, letters, masks, boundary_ids_tensor, device, args.k, args.max_extra)
        for r, ranked in zip(batch, preds):
            results.append((r, ranked))
            if printed < args.print_sample:
                print(f"  ctx=...{r['context'][-40:]!r} letter={r['first letter']!r} "
                      f"answer={r['answer']!r} top{args.k}={ranked!r}")
                printed += 1
        elapsed = time.time() - t0
        done = start + len(batch)
        print(f"[{done}/{len(alpha_rows)} alpha rows] {elapsed:.1f}s elapsed, "
              f"{done/max(elapsed,1e-9):.2f} rows/s", flush=True)

    cat_correct = {1: {}, 5: {}, 10: {}}
    cat_total = {}
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["context", "first_letter", "answer", "category", "top_predictions", "top1", "top5", "top10"])
        for r, ranked in results:
            cat = categorize(r["answer"])
            answer = r["answer"].strip().lower()
            if cat == "number":
                # masked-length scheme, same as report.md's --mask-number check
                candidates = []
                seen = set()
                for w_ in ranked:
                    m = "1" * len(w_.strip())
                    if m not in seen:
                        seen.add(m)
                        candidates.append(m)
            else:
                candidates = [w_.strip().lower() for w_ in ranked]

            cat_total[cat] = cat_total.get(cat, 0) + 1
            hit = {}
            for n in (1, 5, 10):
                ok = answer in candidates[:n]
                hit[n] = ok
                cat_correct[n][cat] = cat_correct[n].get(cat, 0) + ok
            w.writerow([r["context"], r["first letter"], r["answer"], cat, "; ".join(ranked), hit[1], hit[5], hit[10]])

    label = f"{args.model} + LoRA {args.lora}" if args.lora else f"{args.model}, zero-shot"
    print(f"\n--- top-1/5/10 accuracy ({label}, stratified {args.frac*100:.0f}% dev sample) ---")
    for n in (1, 5, 10):
        tot_c = tot_n = 0
        parts = []
        for cat in ("word", "symbol", "number"):
            c, t = cat_correct[n].get(cat, 0), cat_total.get(cat, 0)
            tot_c += c
            tot_n += t
            if t:
                parts.append(f"{cat} {c}/{t} ({c/t*100:.2f}%)")
        print(f"top{n:<2d} overall {tot_c}/{tot_n} ({tot_c/tot_n*100:.2f}%)  |  " + "  ".join(parts))


if __name__ == "__main__":
    main()
