"""Stage 1: runs entirely on the local 4060, zero Kaggle GPU-hours -- per status.md's
"suggestion" section's hardware split. Routes rows, beam-k10 Qwen candidate generation
+ n-gram-top10 union, Qwen's OWN teacher-forced scoring (prefix-cache method, see
score_candidates_prefix_cache.py) + KN5 scoring -- everything EXCEPT Mistral. Writes
one JSON-lines file per input CSV with per-row (context, first_letter, answer,
category, route, candidates, qwen_scores, kn5_scores) -- Stage 2
(kaggle/mistral_only_infer.ipynb, Kaggle T4 x1) reads these files, adds mistral_scores,
and the final blend/report/test-write step combines both by row id.

Reuses infer_topk.py (predict_topk_batch, letter-mask helpers), infer_ngram.py
(load_model, topk_by_letter), blend_ngram.py (KN5Scorer), eval_number_kn5.py
(pick_length), score_candidates_prefix_cache.py (score_candidates_prefix_cache) --
no logic duplicated from those, only orchestration + I/O is new here.

Usage:
    python3 run_qwen_local.py --dev ../data/dev_set_final.csv \
        --test ../data/test_set_no_answer_final.csv --out-dir ../weights/stage1 \
        --dev-subsample 20000
    python3 run_qwen_local.py --demo   # tiny smoke test, --limit 20, no subsample
"""
import argparse
import csv
import json
import random
import re
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_number_kn5 import pick_length
from infer_ngram import load_model, topk_by_letter
from infer_topk import (FirstTokenLetterMask, StopAtWordBoundary, build_letter_masks,
                         detect_boundary, get_boundary_ids, predict_topk_batch)
from blend_ngram import KN5Scorer
from score_candidates_prefix_cache import score_candidates_prefix_cache

CAT_NUMBER = re.compile(r"[0-9]+")
CAT_WORD = re.compile(r"(?=.*[a-z])[a-z']+", re.IGNORECASE)


def categorize(tok):
    if CAT_NUMBER.fullmatch(tok):
        return "number"
    if CAT_WORD.fullmatch(tok):
        return "word"
    return "symbol"


def route(first_letter):
    """Same production routing as the notebooks -- first_letter's OWN character
    class, not categorize(answer) (test has no answer)."""
    if first_letter.isalpha():
        return "word"
    if first_letter.isdigit():
        return "number"
    return "symbol"


def load_rows(path, limit=None, has_answer=True):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    out = []
    for r in rows:
        d = {"context": r["context"], "first_letter": r["first letter"]}
        if has_answer:
            d["answer"] = r["answer"]
        out.append(d)
    return out


def stratified_subsample(rows, target_n, seed=123):
    """Same logic as the v2 notebook's dev subsampling -- proportional per
    categorize(answer) group, +-0.6pt at 95% CI for n=20000 vs the full 94488."""
    rng = random.Random(seed)
    by_cat = {}
    for i, r in enumerate(rows):
        by_cat.setdefault(categorize(r["answer"]), []).append(i)
    selected = []
    for cat, idx in by_cat.items():
        idx = idx[:]
        rng.shuffle(idx)
        take = round(target_n * len(idx) / len(rows))
        selected.extend(idx[:take])
    rng.shuffle(selected)
    return [rows[i] for i in selected]


def run_stage1(rows, model, tok, masks, boundary_ids, device, n, counts, vocab,
                id_to_tok, kn_scorer, number_scorer, beam_k, ngram_topk, gen_batch_size):
    t0 = time.time()
    out_rows = [None] * len(rows)
    word_idx = []
    for i, r in enumerate(rows):
        cat = route(r["first_letter"])
        if cat == "symbol":
            out_rows[i] = {**r, "route": "symbol", "pred": r["first_letter"]}
        elif cat == "number":
            pred = "1" * pick_length(number_scorer, r["context"].split())
            out_rows[i] = {**r, "route": "number", "pred": pred}
        else:
            word_idx.append(i)
    print(f"{len(word_idx)}/{len(rows)} rows routed to word pipeline", flush=True)

    beam_cands = [None] * len(rows)
    for start in range(0, len(word_idx), gen_batch_size):
        batch_idx = word_idx[start:start + gen_batch_size]
        contexts = [rows[i]["context"] for i in batch_idx]
        letters = [rows[i]["first_letter"] for i in batch_idx]
        preds = predict_topk_batch(model, tok, contexts, letters, masks, boundary_ids, device, beam_k)
        for i, p in zip(batch_idx, preds):
            beam_cands[i] = p
        done = start + len(batch_idx)
        if done % (gen_batch_size * 10) == 0 or done == len(word_idx):
            el = time.time() - t0
            print(f"[gen {done}/{len(word_idx)}] {el:.1f}s, {done/max(el,1e-9):.2f} rows/s", flush=True)

    t1 = time.time()
    for j, i in enumerate(word_idx):
        r = rows[i]
        ctx_tokens = r["context"].split()
        ngram_cands = topk_by_letter(ctx_tokens, r["first_letter"], n, counts, vocab, id_to_tok, ngram_topk)
        cands = list(dict.fromkeys((beam_cands[i] or []) + ngram_cands))
        if not cands:
            out_rows[i] = {**r, "route": "word", "candidates": [], "qwen_scores": [], "kn5_scores": []}
            continue
        q_scores = score_candidates_prefix_cache(model, tok, r["context"], cands, device)
        kn5_scores = [kn_scorer.score(ctx_tokens, w) for w in cands]
        out_rows[i] = {**r, "route": "word", "candidates": cands, "qwen_scores": q_scores, "kn5_scores": kn5_scores}
        if (j + 1) % 500 == 0 or j + 1 == len(word_idx):
            el = time.time() - t1
            print(f"[qwen-tf {j+1}/{len(word_idx)}] {el:.1f}s, {(j+1)/max(el,1e-9):.2f} rows/s", flush=True)

    print(f"stage1 done: {len(rows)} rows in {time.time()-t0:.1f}s", flush=True)
    return out_rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def demo():
    """Tiny local smoke test -- 20 rows, no GPU-hour-sized run, catches wiring bugs
    (import errors, signature mismatches) before pointing at the real dev/test files."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    dev_path = os.path.join(here, "..", "data", "dev_set_final.csv")
    assert os.path.exists(dev_path), f"demo needs {dev_path}"
    rows = load_rows(dev_path, limit=20, has_answer=True)
    print(f"loaded {len(rows)} rows for demo -- run without --demo for the real thing")
    print("demo ok (import/wiring check only, no model loaded)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="../data/dev_set_final.csv")
    ap.add_argument("--test", default="../data/test_set_no_answer_final.csv")
    ap.add_argument("--qwen-base", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--qwen-lora", default="../weights/qwen3b_lora_kaggle_6h")
    ap.add_argument("--ngram", default="../weights/ngram_4_a.bin")
    ap.add_argument("--kn-model", default="../weights/kn5.binary")
    ap.add_argument("--number-model", default="../weights/kn5_numbers.binary")
    ap.add_argument("--out-dir", default="../weights/stage1")
    ap.add_argument("--dev-subsample", type=int, default=20000)
    ap.add_argument("--beam-k", type=int, default=10)
    ap.add_argument("--ngram-topk", type=int, default=10)
    ap.add_argument("--gen-batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test row cap")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    n, counts, vocab, id_to_tok = load_model(args.ngram)
    kn_scorer = KN5Scorer(args.kn_model)
    number_scorer = KN5Scorer(args.number_model)
    print("n-gram + KN5 models loaded (CPU)")

    tok = AutoTokenizer.from_pretrained(args.qwen_lora)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    device = args.device if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.qwen_base, dtype=torch.float16, attn_implementation="sdpa").to(device)
    model = PeftModel.from_pretrained(model, args.qwen_lora)
    model.eval()
    print(f"Qwen loaded fp16/sdpa on {device} in {time.time()-t0:.1f}s")

    vocab_size = model.get_output_embeddings().weight.shape[0]
    boundary = detect_boundary(tok)
    masks = build_letter_masks(tok, boundary, device, vocab_size)
    boundary_ids = torch.tensor(get_boundary_ids(tok, boundary), device=device)
    print("boundary marker:", repr(boundary))

    dev_rows = load_rows(args.dev, args.limit, has_answer=True)
    if args.limit is None and args.dev_subsample:
        full_n = len(dev_rows)
        dev_rows = stratified_subsample(dev_rows, args.dev_subsample)
        print(f"dev subsampled: {full_n} -> {len(dev_rows)} rows")
    test_rows = load_rows(args.test, args.limit, has_answer=False)
    print(f"{len(dev_rows)} dev rows, {len(test_rows)} test rows")

    print("\n=== dev ===")
    dev_out = run_stage1(dev_rows, model, tok, masks, boundary_ids, device, n, counts,
                          vocab, id_to_tok, kn_scorer, number_scorer, args.beam_k,
                          args.ngram_topk, args.gen_batch_size)
    write_jsonl(f"{args.out_dir}/dev_qwen_scores.jsonl", dev_out)

    print("\n=== test ===")
    test_out = run_stage1(test_rows, model, tok, masks, boundary_ids, device, n, counts,
                           vocab, id_to_tok, kn_scorer, number_scorer, args.beam_k,
                           args.ngram_topk, args.gen_batch_size)
    write_jsonl(f"{args.out_dir}/test_qwen_scores.jsonl", test_out)

    print(f"\nwrote {args.out_dir}/dev_qwen_scores.jsonl ({len(dev_out)} rows), "
          f"{args.out_dir}/test_qwen_scores.jsonl ({len(test_out)} rows)")


if __name__ == "__main__":
    main()
