"""Cross-encoder reranker for Qwen's top-5 beam, trained on top of the current model
(not replacing it) -- see report.md's "MiniLM rerank ... hurts" section for why a
frozen cosine-similarity bi-encoder failed (no context<->candidate interaction) and
why a trained cross-encoder is the fix.

Two phases, run separately:

  1. gen-data: sample (context, letter, answer) triples from a train_final.src.tok
     chunk NOT used by qwen3b_lora_kaggle's training run (default --skip-lines points
     past both training sessions' consumed range), run them through the current LoRA
     model's beam search (reusing infer_topk.py's exact letter-mask/beam machinery),
     and record each row's top-5 candidates + which index (if any) is the true answer.
     Scope: word-category only -- symbol is already 98.76% (nothing to rerank) and
     number needs a different fix entirely (see report.md), not reranking.

  2. train: fine-tune MiniLM (AutoModelForSequenceClassification, num_labels=1) as a
     listwise cross-encoder over the up-to-5 candidates per row: score each
     (context, candidate) pair, softmax over the row's real (non-padding) candidates,
     cross-entropy against the true index. Rows where the answer wasn't in the beam
     have no valid index and are dropped -- a reranker can only reorder what's already
     there, it can't invent a 6th candidate (top5 ceiling stays the ceiling).

  3. eval: score a trained reranker against dev_predictions_topk5.csv -- the actual
     held-out dev beam output the 69.72%/84.02% baseline/ceiling numbers came from.
     Never used in training (gen-data reads from train_final.src.tok only), so this is
     a clean apples-to-apples comparison with no extra beam-search rerun needed.

Pilot workflow (cheap go/no-go before committing to a full --n 20000 run): gen-data
with a small --n, train on it, eval against dev_predictions_topk5.csv. Crosses the
69.72% baseline meaningfully -> scale --n up and retrain for real. Doesn't -> stop,
it's a negative result same as the MiniLM-cosine one in report.md.

Usage:
    python3 train_reranker.py gen-data --lora ../weights/qwen3b_lora_kaggle \
        --n 20000 --out ../data/reranker_train.csv
    python3 train_reranker.py train --data ../data/reranker_train.csv \
        --out ../weights/minilm_reranker
    python3 train_reranker.py eval --reranker ../weights/minilm_reranker \
        --data ../weights/dev_predictions_topk5.csv
"""
import argparse
import csv
import random

import torch

from infer_topk import (
    build_letter_masks,
    detect_boundary,
    get_boundary_ids,
    predict_topk_batch,
)

RERANKER_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
QWEN_MODEL_ID = "Qwen/Qwen2.5-3B"


def sample_pairs(lines, n, seed, min_ctx_words=3):
    """One (context, letter, answer) triple per sentence, split point chosen at
    random -- mirrors dev_set_final.csv's construction (arbitrary mid-sentence cut,
    not always the last word; verified by inspection, see conversation). Word-category
    only: answer must be alphabetic."""
    rng = random.Random(seed)
    rng.shuffle(lines)
    out = []
    for line in lines:
        words = line.split()
        if len(words) < min_ctx_words + 1:
            continue
        i = rng.randint(min_ctx_words, len(words) - 1)
        answer = words[i]
        if not answer.isalpha():
            continue
        out.append((" ".join(words[:i]), answer[0], answer))
        if len(out) >= n:
            break
    return out


def build_targets(cand_lists, answers, k):
    """Pad each row's candidate list to k (mask=False for padding), find the true
    answer's index if present. Returns (padded_candidates, mask[N,k] bool,
    target_idx[N] long, -1 = answer not in beam -> caller drops that row)."""
    padded, mask, targets = [], [], []
    for cands, ans in zip(cand_lists, answers):
        cands = cands[:k]
        idx = next((j for j, c in enumerate(cands) if c.strip().lower() == ans.strip().lower()), -1)
        row_mask = [True] * len(cands) + [False] * (k - len(cands))
        row_cands = cands + [""] * (k - len(cands))
        padded.append(row_cands)
        mask.append(row_mask)
        targets.append(idx)
    return padded, mask, targets


def pick_best(cands, scores):
    """argmax candidate by score -- eval's rerank step, no padding needed since each
    row's real (non-empty) candidate count is already the array length."""
    return cands[max(range(len(cands)), key=lambda j: scores[j])]


def demo():
    """Self-check: split sampling + padding/target logic, no model/network needed."""
    lines = ["the quick brown fox jumps over", "a b c d e f g h"]
    pairs = sample_pairs(lines, n=10, seed=0, min_ctx_words=3)
    assert all(a.isalpha() for _, _, a in pairs)
    assert all(len(c.split()) >= 3 for c, _, _ in pairs)

    cands = [["cat", "dog"], ["fox", "fix", "fax"], []]
    answers = ["dog", "wolf", "z"]
    padded, mask, targets = build_targets(cands, answers, k=3)
    assert targets == [1, -1, -1]
    assert mask == [[True, True, False], [True, True, True], [False, False, False]]
    assert padded[0] == ["cat", "dog", ""]

    assert pick_best(["cat", "dog", "fox"], [0.1, 0.9, 0.3]) == "dog"
    print("demo ok")


def gen_data(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.lora or args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    boundary = detect_boundary(tokenizer)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto")
    if args.lora:
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()
    model.generation_config.max_length = None

    vocab_size = model.get_output_embeddings().weight.shape[0]
    masks = build_letter_masks(tokenizer, boundary, device, vocab_size)
    boundary_ids_tensor = torch.tensor(get_boundary_ids(tokenizer, boundary), device=device)

    with open(args.train, encoding="utf-8") as f:
        for _ in range(args.skip_lines):
            next(f, None)
        lines = [next(f, "").strip() for _ in range(args.scan_lines)]
    lines = [l for l in lines if l]

    triples = sample_pairs(lines, args.n, args.seed)
    print(f"sampled {len(triples)} word-category triples from {len(lines)} scanned lines")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["context", "first_letter", "answer", "top_predictions"])
        for start in range(0, len(triples), args.batch_size):
            batch = triples[start : start + args.batch_size]
            contexts = [c for c, _, _ in batch]
            letters = [l for _, l, _ in batch]
            preds = predict_topk_batch(model, tokenizer, contexts, letters, masks, boundary_ids_tensor, device, args.k)
            for (c, l, a), ranked in zip(batch, preds):
                w.writerow([c, l, a, "; ".join(ranked)])
            print(f"[{start + len(batch)}/{len(triples)}]", flush=True)


def train(args):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = list(csv.DictReader(open(args.data, encoding="utf-8")))
    cand_lists = [[c.strip() for c in r["top_predictions"].split(";") if c.strip()] for r in rows]
    contexts = [r["context"] for r in rows]
    answers = [r["answer"] for r in rows]

    padded, mask, targets = build_targets(cand_lists, answers, args.k)
    keep = [i for i, t in enumerate(targets) if t != -1]  # answer not in beam -> nothing to rerank, drop
    print(f"{len(keep)}/{len(rows)} rows have the answer in-beam -- training on those")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_ID, num_labels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        rng.shuffle(keep)
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(keep), args.batch_size):
            idxs = keep[start : start + args.batch_size]
            # flatten (batch, k) candidate pairs into one encode call
            pair_ctx, pair_cand = [], []
            for i in idxs:
                for c in padded[i]:
                    pair_ctx.append(contexts[i])
                    pair_cand.append(c if c else "[none]")
            enc = tokenizer(pair_ctx, pair_cand, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            logits = model(**enc).logits.view(len(idxs), args.k)  # (batch, k)

            row_mask = torch.tensor([mask[i] for i in idxs], device=device)
            logits = logits.masked_fill(~row_mask, float("-inf"))
            target = torch.tensor([targets[i] for i in idxs], device=device)

            loss = torch.nn.functional.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch}: avg loss {total_loss / max(n_batches, 1):.4f}")

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved reranker to {args.out}")


def eval_reranker(args):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.reranker)
    model = AutoModelForSequenceClassification.from_pretrained(args.reranker).to(device).eval()

    rows = [r for r in csv.DictReader(open(args.data, encoding="utf-8")) if r["category"] == "word"]
    contexts = [r["context"] for r in rows]
    cand_lists = [[c.strip() for c in r["top_predictions"].split(";") if c.strip()] for r in rows]
    answers = [r["answer"].strip().lower() for r in rows]

    orig_top1 = sum(cands[0].strip().lower() == a for cands, a in zip(cand_lists, answers) if cands)
    in_beam = sum(a in [c.strip().lower() for c in cands] for cands, a in zip(cand_lists, answers))

    # flatten (row, candidate) pairs for one batched scoring pass
    flat_idx, pair_ctx, pair_cand = [], [], []
    for i, (ctx, cands) in enumerate(zip(contexts, cand_lists)):
        for c in cands:
            flat_idx.append(i)
            pair_ctx.append(ctx)
            pair_cand.append(c)

    scores = [None] * len(pair_ctx)
    with torch.inference_mode():
        for start in range(0, len(pair_ctx), args.batch_size):
            enc = tokenizer(pair_ctx[start : start + args.batch_size], pair_cand[start : start + args.batch_size],
                             padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            out = model(**enc).logits.squeeze(-1).tolist()
            scores[start : start + len(out)] = out

    row_scores = [[] for _ in rows]
    for i, s in zip(flat_idx, scores):
        row_scores[i].append(s)

    rerank_top1 = 0
    for cands, sc, a in zip(cand_lists, row_scores, answers):
        if cands and pick_best(cands, sc).strip().lower() == a:
            rerank_top1 += 1

    n = len(rows)
    print(f"n={n} word rows (top5 ceiling / answer-in-beam: {in_beam}/{n} = {in_beam/n*100:.2f}%)")
    print(f"beam log-prob top1:  {orig_top1}/{n} = {orig_top1/n*100:.2f}%")
    print(f"reranked top1:       {rerank_top1}/{n} = {rerank_top1/n*100:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    g = sub.add_parser("gen-data")
    g.add_argument("--model", default=QWEN_MODEL_ID)
    g.add_argument("--lora", default=None)
    g.add_argument("--train", default="../data/train_final.src.tok")
    g.add_argument("--out", default="../data/reranker_train.csv")
    g.add_argument("--skip-lines", type=int, default=3_000_000, help="past both training sessions' consumed chunk -- avoid training the reranker on lines Qwen was already fine-tuned on")
    g.add_argument("--scan-lines", type=int, default=200_000, help="raw lines to scan for candidates before sampling --n from them")
    g.add_argument("--n", type=int, default=20_000)
    g.add_argument("--k", type=int, default=5)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--seed", type=int, default=42)

    t = sub.add_parser("train")
    t.add_argument("--data", default="../data/reranker_train.csv")
    t.add_argument("--out", default="../weights/minilm_reranker")
    t.add_argument("--k", type=int, default=5)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--seed", type=int, default=42)

    e = sub.add_parser("eval")
    e.add_argument("--reranker", default="../weights/minilm_reranker")
    e.add_argument("--data", default="../weights/dev_predictions_topk5.csv")
    e.add_argument("--batch-size", type=int, default=64)

    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo()
        return
    if args.cmd == "gen-data":
        gen_data(args)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        eval_reranker(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
