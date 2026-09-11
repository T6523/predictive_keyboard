"""Zero-shot Mistral-7B-v0.1 on dev_set_final.csv: given context + first letter, predict
the word -- no fine-tuning, base model only. Measures the ceiling before spending a
GPU-day on training (TODO.md, "Suggestion" step 1).

Symbols separated out: the model is only queried when the given first letter is
alphanumeric. A non-alnum first letter (punctuation) is predicted trivially as the
letter itself -- same rule as infer_ngram.py, EDA-justified ~99.6% ceiling, and no
tokenizer piece starts with a punctuation char the way word pieces do, so routing those
through the model would just waste a forward pass to get the same answer a rule gives
for free.

Tokenization-boundary correctness (TODO.md Suggestion item 3): the letter constraint is
applied to the first SUBWORD piece of the next word (SentencePiece "▁"-prefixed --
LLaMA/Mistral's word-boundary marker), not by string-matching decoded text. Greedy
continuation then runs unconstrained, batched round by round, until each row's next
predicted piece is itself a new word-boundary piece (or EOS), at which point that row
freezes and its assembled pieces so far are the prediction. Hand-check a handful of rows
(printed for --limit runs) against the CSV before trusting the score.

Usage:
    python3 infer_transformer.py --dev ../data/dev_set_final.csv \
        --out ../weights/dev_predictions_zeroshot.csv [--limit 2000] [--batch-size 16]
"""
import argparse
import csv
import pickle
import random
import re
import time
from collections import defaultdict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    StoppingCriteria,
)

MODEL_ID = "mistralai/Mistral-7B-v0.1"


def detect_boundary(tokenizer):
    """Word-start marker differs by tokenizer family: SentencePiece (Mistral/LLaMA) uses
    "▁", GPT2-style byte-level BPE (Qwen2.5, GPT2) uses "Ġ". Pick whichever prefixes more
    vocab pieces so this script works across model families without a --model-specific flag.
    """
    vocab = tokenizer.get_vocab()
    counts = {"▁": 0, "Ġ": 0}
    for piece in vocab:
        if piece[:1] in counts:
            counts[piece[:1]] += 1
    return max(counts, key=counts.get)
CAT_NUMBER = re.compile(r"[0-9]+")
CAT_WORD = re.compile(r"(?=.*[a-z])[a-z']+", re.IGNORECASE)  # >=1 real letter -- a bare "'" isn't a word


def categorize(tok):
    if CAT_NUMBER.fullmatch(tok):
        return "number"
    if CAT_WORD.fullmatch(tok):
        return "word"
    return "symbol"


def build_letter_masks(tokenizer, boundary, device, vocab_size):
    """One boolean mask per alphanumeric first-letter: vocab pieces starting with
    boundary + that letter, case-insensitive. Built once, reused across all rows.

    vocab_size = model's logits width, not len(tokenizer) -- some models (e.g. Qwen2.5)
    pad the embedding matrix past the tokenizer's actual vocab, so scores.shape[-1] can
    be bigger than len(tokenizer); mask must match scores' width or masked_fill crashes.
    """
    vocab = tokenizer.get_vocab()
    letters = list("abcdefghijklmnopqrstuvwxyz0123456789")
    masks = {c: torch.zeros(vocab_size, dtype=torch.bool) for c in letters}
    for piece, idx in vocab.items():
        if len(piece) > 1 and piece[0] == boundary and piece[1].lower() in masks:
            masks[piece[1].lower()][idx] = True
    return {c: m.to(device) for c, m in masks.items()}


def load_ngram_number_predictor(path):
    """Mirrors infer_ngram.py's predict_word: backs off from n-gram context down to a
    unigram best-per-letter fallback. Used only for number rows here -- the digit "1"
    is always the given first letter (see the number-anonymization comment below), and
    the n-gram beats a flat length guess by conditioning length on context (71.68% vs
    39.6% on dev, measured) -- an ensemble win, not a model-capability one.
    """
    with open(path, "rb") as f:
        m = pickle.load(f)
    n, counts, vocab, id_to_tok = m["n"], m["counts"], m["vocab"], m["id_to_tok"]

    letter_ids = defaultdict(list)
    for tok, i in vocab.items():
        if tok in ("<s>", "</s>") or not tok:
            continue
        letter_ids[tok[0]].append(i)
    unigram = counts[0][()]
    unigram_best = {
        letter: max(ids, key=lambda i: unigram.get(i, 0)) for letter, ids in letter_ids.items()
    }

    def best_by_letter(d, letter):
        best_id, best_c = None, -1
        for wid, c in d.items():
            tok = id_to_tok[wid]
            if tok and tok[0] == letter and c > best_c:
                best_id, best_c = wid, c
        return best_id

    def predict(context_tokens, letter):
        ids = [vocab.get(t) for t in context_tokens[-(n - 1):]]
        for j in range(len(ids), 0, -1):
            ctx_ids = ids[-j:]
            if None in ctx_ids:
                continue
            d = counts[j].get(tuple(ctx_ids))
            if d:
                cand = best_by_letter(d, letter)
                if cand is not None:
                    return id_to_tok[cand]
        cand = unigram_best.get(letter)
        return id_to_tok[cand] if cand is not None else "1" * 2  # "11" fallback

    return predict


def get_boundary_ids(tokenizer, boundary):
    """All vocab ids whose piece starts a new word (boundary-prefixed). Passed to
    generate() as extra stop tokens, so its native per-row early-stop (with KV-cache)
    halts each row right as the next word begins -- no custom StoppingCriteria needed.
    """
    return [idx for piece, idx in tokenizer.get_vocab().items() if piece.startswith(boundary)]


class FirstTokenLetterMask(LogitsProcessor):
    """Constrains only the FIRST generated token per row to that row's letter-matching
    vocab pieces; every later step passes through unconstrained -- same semantics as
    the old manual loop, just detected by prompt-length instead of a round counter.
    """

    def __init__(self, letter_mask, prompt_len):
        self.letter_mask = letter_mask
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] == self.prompt_len:
            scores = scores.masked_fill(~self.letter_mask, float("-inf"))
        return scores


class StopAtWordBoundary(StoppingCriteria):
    """Per-row stop: fires once the LAST generated token is boundary-tagged (a new word
    started) or real EOS -- but never at the forced first step, since that token is
    boundary-tagged by construction (the letter constraint requires it) and firing
    there would halt every row before any continuation happens at all.
    """

    def __init__(self, prompt_len, boundary_ids_tensor, eos_id):
        self.prompt_len = prompt_len
        self.boundary_ids_tensor = boundary_ids_tensor
        self.eos_id = eos_id

    def __call__(self, input_ids, scores, **kwargs):
        cur_len = input_ids.shape[1]
        if cur_len <= self.prompt_len + 1:  # only the forced first token generated so far
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        last = input_ids[:, -1]
        return torch.isin(last, self.boundary_ids_tensor) | (last == self.eos_id)


@torch.inference_mode()
def predict_batch(model, tokenizer, contexts, letters, masks, boundary_ids_tensor, device, max_extra=4):
    """Constrained-first-piece + KV-cached greedy continuation via model.generate().
    Replaces the old full-recompute-per-round loop: cache means each step only costs
    1 new token's attention, not the whole growing sequence.
    """
    enc = tokenizer(contexts, return_tensors="pt", padding=True).to(device)
    letter_mask = torch.stack([masks[l.lower()] for l in letters])  # (B, vocab)
    prompt_len = enc["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id

    out = model.generate(
        **enc,
        max_new_tokens=max_extra + 1,
        do_sample=False,
        logits_processor=[FirstTokenLetterMask(letter_mask, prompt_len)],
        stopping_criteria=[StopAtWordBoundary(prompt_len, boundary_ids_tensor, eos_id)],
        pad_token_id=eos_id,
    )
    generated = out[:, prompt_len:].tolist()

    boundary_set = set(boundary_ids_tensor.tolist())
    words = []
    for row in generated:
        # position 0 is the forced first piece -- always boundary-tagged by
        # construction, never cut it. Look for a stop token from position 1 onward
        # (start of the NEXT word, or real EOS/pad).
        cut = next((i for i, tid in enumerate(row) if i > 0 and (tid in boundary_set or tid == eos_id)), len(row))
        words.append(tokenizer.decode(row[:cut]).strip())
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--dev", default="../data/dev_set_final.csv")
    ap.add_argument("--out", default="../weights/dev_predictions_zeroshot.csv")
    ap.add_argument("--limit", type=int, default=0, help="0 = full dev set")
    ap.add_argument("--seed", type=int, default=0, help="random sample seed, only used with --limit")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-extra", type=int, default=4, help="max continuation pieces after the first")
    ap.add_argument("--print-sample", type=int, default=10, help="print first N alnum predictions for hand-check")
    ap.add_argument("--ngram-model", default=None, help="if set, use this n-gram model for number-category rows instead of the flat '11' guess")
    ap.add_argument("--mask-number", action="store_true", help="route number rows through the model like alpha rows, then mask the prediction to '1'*len before scoring -- checks length prediction only, no ngram, no ensemble")
    ap.add_argument("--lora", default=None, help="if set, load this PEFT/LoRA adapter dir on top of --model (e.g. a Kaggle-trained checkpoint) instead of running the bare base model")
    args = ap.parse_args()

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
    model.generation_config.max_length = None  # silence the harmless but noisy "both max_new_tokens
                                                # and max_length are set" warning generate() prints
                                                # every call -- max_new_tokens already wins either way

    vocab_size = model.get_output_embeddings().weight.shape[0]
    masks = build_letter_masks(tokenizer, boundary, device, vocab_size)
    boundary_ids_tensor = torch.tensor(get_boundary_ids(tokenizer, boundary), device=device)

    with open(args.dev, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = random.Random(args.seed).sample(rows, min(args.limit, len(rows)))

    results = []
    # Numbers in this dataset are anonymized: every number answer is the digit "1"
    # repeated to the original number's length ("1234" -> "1111"), so first letter is
    # always "1" -- a real word never starts with a digit, so a digit first-letter is
    # an unambiguous number-placeholder signal, same trick as the symbol rule below.
    # No LLM ever generates "1111" (not real text), hence the flat 0% number accuracy
    # seen across every zero-shot model tried -- skip the model entirely here too.
    # Length is genuinely ambiguous without context (dev: 1-digit 30.7%, 2-digit 39.6%,
    # 3-digit 18.4%, 4-digit 11.3%) -- majority-class "11" is the free win; the n-gram
    # baseline beats this (71.68%) by conditioning length on context, an ensemble
    # opportunity (TODO.md Suggestion item 5), not implemented here.
    alpha_rows = [r for r in rows if r["first letter"].isalpha()]
    digit_rows = [r for r in rows if r["first letter"].isdigit()]
    symbol_rows = [r for r in rows if not r["first letter"].isalnum()]

    for r in symbol_rows:
        results.append((r, r["first letter"]))

    if args.mask_number:
        # --mask-number: don't shortcut digit rows at all -- run them through the
        # model exactly like alpha rows (below), scoring masks the prediction to
        # '1'*len afterward. No ngram, no ensemble.
        alpha_rows = alpha_rows + digit_rows
    else:
        ngram_predict = load_ngram_number_predictor(args.ngram_model) if args.ngram_model else None
        for r in digit_rows:
            pred = ngram_predict(r["context"].split(), r["first letter"]) if ngram_predict else "11"
            results.append((r, pred))

    t0 = time.time()
    printed = 0
    for start in range(0, len(alpha_rows), args.batch_size):
        batch = alpha_rows[start : start + args.batch_size]
        contexts = [r["context"] for r in batch]
        letters = [r["first letter"] for r in batch]
        preds = predict_batch(model, tokenizer, contexts, letters, masks, boundary_ids_tensor, device, args.max_extra)
        for r, p in zip(batch, preds):
            results.append((r, p))
            if printed < args.print_sample:
                print(f"  ctx=...{r['context'][-40:]!r} letter={r['first letter']!r} "
                      f"answer={r['answer']!r} pred={p!r}")
                printed += 1
        elapsed = time.time() - t0
        done = start + len(batch)
        print(f"[{done}/{len(alpha_rows)} alpha rows] {elapsed:.1f}s elapsed, "
              f"{done/max(elapsed,1e-9):.2f} rows/s", flush=True)

    cat_correct, cat_total = {}, {}
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["context", "first_letter", "answer", "category", "prediction", "correct"])
        for r, pred in results:
            cat = categorize(r["answer"])
            scored_pred = pred.strip()
            if args.mask_number and cat == "number":
                scored_pred = "1" * len(scored_pred)  # length-only check, digits don't matter
            correct = scored_pred.lower() == r["answer"].strip().lower()
            cat_total[cat] = cat_total.get(cat, 0) + 1
            cat_correct[cat] = cat_correct.get(cat, 0) + correct
            w.writerow([r["context"], r["first letter"], r["answer"], cat, pred, correct])

    label = f"{args.model} + LoRA {args.lora}" if args.lora else f"{args.model}, zero-shot no fine-tune"
    print(f"\n--- accuracy ({label}) ---")
    tot_c = tot_n = 0
    for cat in ("word", "symbol", "number"):
        c, n = cat_correct.get(cat, 0), cat_total.get(cat, 0)
        tot_c += c
        tot_n += n
        if n:
            print(f"{cat:8s} {c:6d} / {n:6d}  {c/n*100:.2f}%")
    print(f"{'overall':8s} {tot_c:6d} / {tot_n:6d}  {tot_c/tot_n*100:.2f}%")


if __name__ == "__main__":
    main()
