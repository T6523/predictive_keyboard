#!/usr/bin/env python3
"""Eval-only companion to train_and_eval.py -- loads a previously-saved LoRA adapter (base model
+ adapter merged via plain transformers+peft, see bug #3 below for why not unsloth) instead of
training fresh, then runs the same constrained next-word eval. Split out because the original 8hr
training kernel's own eval pass
OOM'd (model(seqs).logits materialized full (batch, ctx_len+L, vocab) logits regardless of L --
Qwen's ~152k vocab made that blow past a T4's 14.56GB independent of batch size; see the same fix
in train_and_eval.py's best_candidate()). This retries just the eval step against the adapter
that already finished training and saved -- no need to spend another 8 GPU-hours retraining.

First real run (post-OOM-fix) measured ~19s/row -- still unusable (~400hr for the full eval set).
Root cause is NOT the OOM issue: unlike the from-scratch GPT2 (word-level vocab, one masked
forward pass per row), Qwen's subword vocab means one candidate = one multi-token sequence, so
best_candidate() reruns the FULL context through all 24 layers once per batch of candidates --
and a busy letter bucket (vocab.txt's 's' words alone: 10422) needs ~163 batches of 64 per row.
A from-scratch KV-cache (reusing the context's past_key_values across candidate batches) is the
textbook fix but proved fragile here: manually driving DynamicCache/cache_position outside
model.generate() hit a persistent SDPA shape-mismatch error on both GPT2 and Qwen2 test configs
(transformers' cache-position/attention-mask plumbing for this isn't a stable public contract).

Shortlisting: score every same-letter candidate with the cheap in-domain 5-gram (model_a.klm,
already trained, pure-CPU, no GPU/tensor cost) and only send its top SHORTLIST_K to Qwen. Measured
locally on 500 devv_eval rows: K=80 (default) keeps the true answer in the shortlist 94.95% of the
time (K=50 -> 92.3%, K=200 -> 96.7%, K=800 -> 99.1%) -- picked over K=200 to cut per-row Qwen
forward-batches further after K=200 + the KV-cache still measured ~7.7h projected total (0.29s/row
on a T4 without unsloth's fused kernels, see bug #3), too close to Kaggle's session cap for
comfort. ponytail: shortlist ceiling is real (rows where the true answer isn't in the top K are
unwinnable regardless of what Qwen says) -- raise --shortlist-k if eval shows this costing
noticeable accuracy (and runtime allows it).

KV-cache: the shortlist alone (first shipped version of this file) cut candidates 10422->200 for
the worst letter bucket but was STILL ~0.85s/row (full context re-encoded through all 24 layers
once per length-grouped batch of candidates) -- unusably slow at 75860+18965 rows. Fixed by
encoding the context ONCE (same cost as GPT2's single masked forward) and reusing its
past_key_values across every candidate batch, so each batch only pays for its own 1-6 new tokens.

Caught three real bugs building this, all worth naming because they're easy to reintroduce:
  1. DynamicCache.batch_repeat_interleave(B) mutates the cache IN PLACE and returns None -- code
     that does `cache = cache.batch_repeat_interleave(B)` silently ends up passing
     past_key_values=None (verified: produces logits ~0.1 off in log-space from a from-scratch
     recompute, wrong enough to matter, not enough to look obviously broken). Fix: call it as a
     bare statement on a copy.deepcopy() of the base cache, never reassign its return value.
  2. The ORIGINAL (pre-cache) best_candidate() scored logits_to_keep=L against target=candidate
     ids directly -- verified empirically this is off by one position: logits_to_keep=L returns
     the model's last L logit ROWS, which predict tokens L+1..2L relative to context end, not
     tokens 1..L (confirmed even the L=1 single-token case was misaligned). Every candidate score
     the original code ever produced was against the wrong logit row. Never shipped/reported --
     caught while building the cache version, which scores correctly (see best_candidate()) by
     using the context forward's OWN last-position logit for the candidate's first token, and
     shifting target by one for the rest.
  3. FastLanguageModel.for_inference(model) patches the model's forward to route any call with
     past_key_values is not None through unsloth's own single-token decode path (fast_forward_
     inference), which hard-asserts q_len == 1 -- confirmed via a real run's traceback
     (AssertionError at unsloth/models/llama.py's LlamaModel_fast_forward_inference_custom) and
     via unsloth's own GitHub issues (#497: custom past_key_values unsupported; the documented
     workaround is merge_and_unload()). It's built for model.generate()'s one-new-token-at-a-time
     loop, not for feeding L>1 new tokens against a cache in one call, which is exactly what
     candidate scoring needs. Fix: skip unsloth's inference path entirely -- load the base model
     and adapter via plain transformers + peft, merge_and_unload() the LoRA into ordinary weights,
     and run best_candidate() against that (vanilla Qwen2ForCausalLM.forward has none of this
     restriction; verified locally against a from-scratch Qwen2Config before shipping).

Attach four datasets: the LoRA adapter (adapter_config.json + adapter_model.safetensors +
tokenizer.* at its root -- currently a placeholder, upload the real files when ready),
teekn07/keyboard (devv_eval.csv/devv_test.csv), teekn07/predictive-keyboard-vocab (vocab.txt),
teekn07/predictive-keyboard-kenlm-a (model_a.klm).

Usage: python3 eval_qwen.py [--eval-limit 500] [--shortlist-k 200]
"""
import argparse
import copy
import csv
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import peft  # noqa: F401
    from peft.import_utils import is_torchao_available
    is_torchao_available()  # raises if the image's preinstalled torchao is too old for this peft
except ImportError:
    # Kaggle's stock image shipped torchao 0.10.0, but this peft's LoRA dispatcher unconditionally
    # requires >=0.16.0 (checked even though our adapter has nothing to do with torchao) --
    # confirmed via a real run's traceback. Upgrading both together avoids re-discovering whatever
    # older peft/torchao pairing the image happens to ship next.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "peft", "torchao"], check=True)
try:
    import kenlm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kenlm"], check=True)
    import kenlm

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MAX_SEQ_LEN = 1024


# mirrors scripts/symbol_predict.py exactly (see train_and_eval.py's copy for provenance).
def predict_symbol(letter):
    return "[UNK]" if letter == "[" else letter


def is_symbol_letter(letter):
    return not str(letter).isalnum()


def resolve(filename):
    local = ROOT / filename
    if local.exists():
        return local
    hits = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if hits:
        return Path(hits[0])
    raise FileNotFoundError(filename)


def resolve_adapter_dir():
    """The adapter dataset's root is wherever adapter_config.json landed -- same recursive-glob
    reasoning as resolve() (script kernels nest attached datasets under /kaggle/input/datasets/...)."""
    local = ROOT / "adapter"
    if (local / "adapter_config.json").exists():
        return local
    hits = glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
    if hits:
        return Path(hits[0]).parent
    raise FileNotFoundError("adapter_config.json not found under /kaggle/input -- attach the adapter dataset")


def _prime(model, context_tokens, s1, s2):
    """Same as predict_accuracy.py's _prime -- duplicated, this is a standalone script kernel
    (no repo imports). Returns whichever of s1/s2 ends up holding the final context state."""
    model.BeginSentenceWrite(s1)
    for tok in context_tokens:
        model.BaseScore(s1, tok, s2)
        s1, s2 = s2, s1
    return s1


def _batch_repeat_past(past, n):
    """Version-robust past_key_values repeat. Newer transformers returns a DynamicCache, whose
    batch_repeat_interleave(n) mutates in place and returns None (see best_candidate()'s
    docstring). Kaggle's shipped transformers (5.5.0, older than what's installed locally) instead
    returns the legacy tuple-of-(key, value)-per-layer format, which has no such method at all --
    confirmed via `AttributeError: 'tuple' object has no attribute 'batch_repeat_interleave'` on a
    real run. Handle both rather than pin a version."""
    if hasattr(past, "batch_repeat_interleave"):
        past.batch_repeat_interleave(n)
        return past
    return tuple((k.repeat_interleave(n, dim=0), v.repeat_interleave(n, dim=0)) for k, v in past)


def shortlist(kenlm_model, context_tokens, candidates, k):
    """Rank same-letter candidates by the cheap in-domain 5-gram, keep the top k. Cuts Qwen's
    per-row forward-pass count from O(bucket size) to O(k) -- see module docstring for the
    measured recall/speed tradeoff. candidates: list[(word, token_id_tuple)]."""
    if len(candidates) <= k:
        return candidates
    s1, s2 = kenlm.State(), kenlm.State()
    state = _prime(kenlm_model, context_tokens, s1, s2)
    out = kenlm.State()
    scored = [(kenlm_model.BaseFullScore(state, w, out).log_prob, w, ids) for w, ids in candidates]
    scored.sort(reverse=True)
    return [(w, ids) for _, w, ids in scored[:k]]


def load_vocab_by_letter(vocab_path):
    by_letter = {}
    with open(vocab_path, encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w and w.isalnum():
                by_letter.setdefault(w[0], []).append(w)
    return by_letter


def tokenize_vocab(tokenizer, vocab_by_letter):
    words = [w for ws in vocab_by_letter.values() for w in ws]
    enc = tokenizer(words, add_special_tokens=False)["input_ids"]
    return {w: tuple(ids) for w, ids in zip(words, enc)}


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


@torch.no_grad()
def best_candidate(model, device, context_ids, candidates, max_batch=64):
    """See module docstring for the two bugs this replaced (batch_repeat_interleave mutates in
    place / logits_to_keep off-by-one). Encodes context once, reuses its past_key_values across
    every candidate batch -- each batch then only pays for its own 1-6 new tokens instead of
    re-running the full context through all 24 layers per batch."""
    ctx = torch.tensor([context_ids], device=device)
    out = model(ctx, use_cache=True)
    base_past = out.past_key_values
    ctx_last_logp = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
    ctx_len = len(context_ids)

    by_len = {}
    for w, ids in candidates:
        by_len.setdefault(len(ids), []).append((w, ids))

    best_w, best_s = None, float("-inf")
    for L, group in by_len.items():
        for chunk in batched(group, max_batch):
            words, id_lists = zip(*chunk)
            first_term = torch.stack([ctx_last_logp[ids[0]] for ids in id_lists])
            if L == 1:
                score = first_term
            else:
                batch_past = _batch_repeat_past(copy.deepcopy(base_past), len(chunk))
                target = torch.tensor(id_lists, device=device)
                cache_position = torch.arange(ctx_len, ctx_len + L, device=device)
                logits = model(target, past_key_values=batch_past, cache_position=cache_position,
                                use_cache=False).logits
                logp = torch.log_softmax(logits.float(), dim=-1)
                # logp[:, j] predicts target[:, j+1] (teacher-forced next-token, j=0..L-2);
                # target[:, 0]'s own probability is first_term, already scored off the context pass.
                rest = logp[:, :-1, :].gather(-1, target[:, 1:].unsqueeze(-1)).squeeze(-1).sum(dim=-1)
                score = first_term + rest
            i = int(score.argmax())
            if score[i].item() > best_s:
                best_s, best_w = score[i].item(), words[i]
    return best_w


def run_eval(path, model, tokenizer, device, vocab_cache, vocab_by_letter, max_ctx, limit, out_csv,
             kenlm_model, shortlist_k):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    correct, out_rows = 0, []
    t0 = time.time()
    for n, row in enumerate(rows, 1):
        letter = row["first letter"]
        if is_symbol_letter(letter):
            pred = predict_symbol(letter)
        else:
            candidates = [(w, vocab_cache[w]) for w in vocab_by_letter.get(letter.lower(), ())]
            pred = None
            if candidates:
                context_tokens = row["context"].split()
                candidates = shortlist(kenlm_model, context_tokens, candidates, shortlist_k)
                ctx_ids = tokenizer.encode(row["context"], add_special_tokens=False)[-max_ctx:]
                pred = best_candidate(model, device, ctx_ids, candidates)
        is_correct = pred == row["answer"]
        correct += is_correct
        out_rows.append({"history": row["context"], "prefix": letter, "target": row["answer"],
                          "predicted": pred, "is_correct": is_correct})
        if n % 200 == 0:
            print(f"  {n}/{len(rows)} ({time.time() - t0:.0f}s)")
    acc = correct / max(len(rows), 1)
    print(f"{path.name}: accuracy {correct}/{len(rows)} = {acc:.4f}")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["history", "prefix", "target", "predicted", "is_correct"])
        w.writeheader()
        w.writerows(out_rows)
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-limit", type=int, default=None, help="cap rows per eval file (smoke test)")
    ap.add_argument("--shortlist-k", type=int, default=80,
                     help="keep only the top-k 5-gram-ranked candidates per row before scoring with Qwen")
    args = ap.parse_args()

    adapter_dir = resolve_adapter_dir()
    eval_path = resolve("devv_eval.csv")
    test_path = resolve("devv_test.csv")
    vocab_path = resolve("vocab.txt")
    kenlm_path = resolve("model_a.klm")
    kenlm_model = kenlm.Model(str(kenlm_path))

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_name = json.loads((adapter_dir / "adapter_config.json").read_text())["base_model_name_or_path"]
    print(f"--- loading base ({base_name}) + adapter from {adapter_dir}, merging (no unsloth -- see module docstring) ---")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    base = AutoModelForCausalLM.from_pretrained(base_name, dtype=compute_dtype)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    model.eval().to(device)

    vocab_by_letter = load_vocab_by_letter(vocab_path)
    vocab_cache = tokenize_vocab(tokenizer, vocab_by_letter)
    max_ctx = MAX_SEQ_LEN - 8  # room for the longest candidate's tokens

    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else ROOT
    eval_acc = run_eval(eval_path, model, tokenizer, device, vocab_cache, vocab_by_letter, max_ctx,
                         args.eval_limit, out_dir / "devv_eval_predictions.csv", kenlm_model, args.shortlist_k)
    test_acc = run_eval(test_path, model, tokenizer, device, vocab_cache, vocab_by_letter, max_ctx,
                         args.eval_limit, out_dir / "devv_test_predictions.csv", kenlm_model, args.shortlist_k)
    print(f"devv_eval accuracy: {eval_acc:.4f}")
    print(f"devv_test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
