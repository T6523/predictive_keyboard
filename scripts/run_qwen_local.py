"""Stage 1: runs entirely on the local 4060, zero Kaggle GPU-hours -- per status.md's
"suggestion" section's hardware split. Routes rows, beam-k10 Qwen candidate generation
+ n-gram-top10 union, Qwen's OWN teacher-forced scoring (prefix-cache method, see
score_candidates_prefix_cache.py) + KN5 scoring -- everything EXCEPT Mistral. Writes
one JSON-lines file per input CSV with per-row (index, context, first_letter, answer,
category, route, candidates, qwen_scores, kn5_scores) -- Stage 2
(kaggle/mistral_only_infer.ipynb, Kaggle T4 x1) reads these files, adds mistral_scores,
and the final blend/report/test-write step combines both by row id.

Uses the pre-merged Qwen+LoRA model (weights/qwen3b_merged_6h, LoRA baked in via
merge_and_unload -- verified byte-identical predictions to base+LoRA, diff ~1e-3
fp16 noise) -- also faster solo than the separate base+PeftModel load (no LoRA
forward-pass overhead), matching the fix applied on the Kaggle side after its dual-GPU
attempt kept crashing (host RAM ceiling, unrelated to the merge but the merge helps
speed regardless).

--row-frac-start/--row-frac-end let this run process only a SLICE of the (subsampled)
dev + full test sets, concurrently with a Kaggle qwen-only run processing the
complementary slice -- see kaggle/qwen_only_infer.ipynb, which applies the SAME
stratified_subsample (same seed) before its own slicing, so the two slices union to
the full subsampled dev set + full test set with no gaps/overlap.

Reuses infer_topk.py (predict_topk_batch, letter-mask helpers), infer_ngram.py
(load_model, topk_by_letter), blend_ngram.py (KN5Scorer), eval_number_kn5.py
(pick_length), score_candidates_prefix_cache.py (score_candidates_prefix_cache) --
no logic duplicated from those, only orchestration + I/O is new here.

Usage:
    python3 run_qwen_local.py --dev ../data/dev_set_final.csv \
        --test ../data/test_set_no_answer_final.csv --out-dir ../weights/stage1 \
        --dev-subsample 20000 --row-frac-start 0.0 --row-frac-end 0.207
    python3 run_qwen_local.py --demo   # tiny smoke test, --limit 20, no subsample
"""
import argparse
import csv
import json
import random
import re
import time

import torch
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
    """Same logic as the Kaggle worker's dev subsampling -- proportional per
    categorize(answer) group, +-0.6pt at 95% CI for n=20000 vs the full 94488.
    MUST stay byte-identical (same seed, same algorithm) on both sides -- it's the
    only thing that keeps local's and Kaggle's row-frac slices covering the same
    underlying subsampled set."""
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
    """rows: list of row dicts, each already carrying its original-list "index" key
    (added by the caller before slicing) so a partial slice's output can still be
    merged back in the right position later."""
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
    ap.add_argument("--qwen-merged", default="../weights/qwen3b_merged_6h")
    ap.add_argument("--ngram", default="../weights/ngram_4_a.bin")
    ap.add_argument("--kn-model", default="../weights/kn5.binary")
    ap.add_argument("--number-model", default="../weights/kn5_numbers.binary")
    ap.add_argument("--out-dir", default="../weights/stage1")
    ap.add_argument("--dev-subsample", type=int, default=20000)
    ap.add_argument("--row-frac-start", type=float, default=0.0)
    ap.add_argument("--row-frac-end", type=float, default=1.0)
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

    tok = AutoTokenizer.from_pretrained(args.qwen_merged)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    device = args.device if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.qwen_merged, dtype=torch.float16, attn_implementation="sdpa").to(device)
    model.eval()
    print(f"Qwen (merged) loaded fp16/sdpa on {device} in {time.time()-t0:.1f}s")

    vocab_size = model.get_output_embeddings().weight.shape[0]
    boundary = detect_boundary(tok)
    masks = build_letter_masks(tok, boundary, device, vocab_size)
    boundary_ids = torch.tensor(get_boundary_ids(tok, boundary), device=device)
    print("boundary marker:", repr(boundary))

    dev_full = load_rows(args.dev, args.limit, has_answer=True)
    if args.limit is None and args.dev_subsample:
        full_n = len(dev_full)
        dev_full = stratified_subsample(dev_full, args.dev_subsample)
        print(f"dev subsampled: {full_n} -> {len(dev_full)} rows")
    test_full = load_rows(args.test, args.limit, has_answer=False)

    s, e = int(args.row_frac_start * len(dev_full)), int(args.row_frac_end * len(dev_full))
    dev_rows = [{"index": i, **dev_full[i]} for i in range(s, e)]
    s, e = int(args.row_frac_start * len(test_full)), int(args.row_frac_end * len(test_full))
    test_rows = [{"index": i, **test_full[i]} for i in range(s, e)]
    print(f"{len(dev_rows)} dev rows, {len(test_rows)} test rows in this run's slice "
          f"[{args.row_frac_start}, {args.row_frac_end})")

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
