"""Stage 2 worker -- one process, one GPU, one Mistral instance. Two of these run
concurrently (via subprocess, NOT threads -- separate OS processes means separate CUDA
contexts, sidesteps the exact thread-safety issues that crashed the earlier
ThreadPoolExecutor attempt, see status.md's v6-v9 crash log) to use BOTH of Kaggle's
T4x2 GPUs instead of leaving the second one idle while still paying its quota cost.

Scores every row assigned to this shard (index % --num-shards == --shard) via
score_candidates_prefix_cache, writes a compact JSONL of just the assigned rows
({"index": original_row_index, "mistral_scores": [...]})-- the caller (Kaggle
notebook cell) merges both shards' output back into full-length dev/test files
combine_final.py expects.

Usage (normally launched by the notebook, not by hand):
    python3 run_mistral_worker.py --dev-jsonl dev_qwen_scores.jsonl \
        --test-jsonl test_qwen_scores.jsonl --mistral-base ... --device cuda:0 \
        --shard 0 --num-shards 2 --out-dir /kaggle/working
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from score_candidates_prefix_cache import score_candidates_prefix_cache


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def score_shard(rows, model, tok, device, shard, num_shards, name):
    t0 = time.time()
    out = []
    assigned = [i for i in range(len(rows)) if i % num_shards == shard
                and rows[i]["route"] == "word" and rows[i].get("candidates")]
    for j, i in enumerate(assigned):
        r = rows[i]
        scores = score_candidates_prefix_cache(model, tok, r["context"], r["candidates"], device)
        out.append({"index": i, "mistral_scores": scores})
        if (j + 1) % 200 == 0 or j + 1 == len(assigned):
            el = time.time() - t0
            print(f"[{name} shard{shard} {j+1}/{len(assigned)}] {el:.1f}s, {(j+1)/max(el,1e-9):.2f} rows/s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-jsonl", required=True)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--mistral-base", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test row cap per set")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.mistral_base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.mistral_base, dtype=torch.float16, attn_implementation="sdpa").to(args.device)
    model.eval()
    print(f"[shard{args.shard}] Mistral loaded fp16/sdpa on {args.device} in {time.time()-t0:.1f}s", flush=True)

    dev_rows = load_jsonl(args.dev_jsonl)
    test_rows = load_jsonl(args.test_jsonl)
    if args.limit:
        dev_rows, test_rows = dev_rows[:args.limit], test_rows[:args.limit]

    dev_out = score_shard(dev_rows, model, tok, args.device, args.shard, args.num_shards, "dev")
    test_out = score_shard(test_rows, model, tok, args.device, args.shard, args.num_shards, "test")

    with open(f"{args.out_dir}/dev_mistral_shard{args.shard}.jsonl", "w", encoding="utf-8") as f:
        for r in dev_out:
            f.write(json.dumps(r) + "\n")
    with open(f"{args.out_dir}/test_mistral_shard{args.shard}.jsonl", "w", encoding="utf-8") as f:
        for r in test_out:
            f.write(json.dumps(r) + "\n")
    print(f"[shard{args.shard}] wrote {len(dev_out)} dev + {len(test_out)} test scored rows", flush=True)


if __name__ == "__main__":
    main()
