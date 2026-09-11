"""Prefix-KV-cache teacher-forced scoring -- per status.md's "suggestion" section.

Old method (score_candidates_batch_multi in the notebook / score_candidates.py):
reprocesses the FULL context + candidate for every candidate. ~30-token context x
~18 candidates/row = ~600 token-forwards/row/model.

This method: forward the context ONCE (use_cache=True), expand the resulting cache
across the candidate batch dimension, then forward only the (short, padded)
candidate continuations with that shared cache. ~30 + ~(18 x candidate_len) token-
forwards/row/model instead -- the ~10x cut status.md's suggestion estimated.

Same log-prob decomposition as the old method, just split across two forward calls:
  token 0 of each candidate <- last logit of the context-only forward
  token i>0 of each candidate <- candidate-forward's logits[:, i-1]

Kept as a SEPARATE file from score_candidates.py / the notebook's
score_candidates_batch_multi (not overwritten) per user request -- swap in once
verified on the real models, old code stays as a rollback.

Usage (demo, no GPU needed -- CPU correctness check against the old full-reprocess
method on a tiny real model):
    python3 score_candidates_prefix_cache.py --demo
"""
import argparse

import torch


def _expand_cache(past_key_values, n):
    """Repeat a just-built Cache along the batch dim so it can be reused for n
    candidates in one forward call. Tries the named Cache API method first
    (transformers>=4.5x-ish); falls back to manual per-layer tensor repeat for older/
    newer versions where the method name or Cache internals differ -- Kaggle's
    preinstalled transformers version is not guaranteed to match local."""
    if hasattr(past_key_values, "batch_repeat_interleave"):
        past_key_values.batch_repeat_interleave(n)
        return past_key_values
    # fallback: legacy tuple-of-(key, value) format, or a Cache exposing
    # key_cache/value_cache lists directly
    if hasattr(past_key_values, "key_cache"):
        for i in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[i] = past_key_values.key_cache[i].repeat_interleave(n, dim=0)
            past_key_values.value_cache[i] = past_key_values.value_cache[i].repeat_interleave(n, dim=0)
        return past_key_values
    return tuple((k.repeat_interleave(n, dim=0), v.repeat_interleave(n, dim=0)) for k, v in past_key_values)


@torch.inference_mode()
def score_candidates_prefix_cache(model, tokenizer, context, candidates, device):
    """Teacher-forced full-word log-prob for every candidate of ONE row, sharing one
    context forward pass via a prefix KV cache. Numerically equivalent to
    score_candidates_batch's full-reprocess method (verified by demo() below) --
    same per-token log-probs, just computed via two forwards instead of one full
    reprocess per candidate.

    context: string. candidates: list of candidate words.
    Returns: list of float log-probs, parallel to candidates.
    """
    if not candidates:
        return []

    ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    cand_id_lists = [tokenizer(" " + cand, add_special_tokens=False)["input_ids"] for cand in candidates]
    # NOTE: " "+cand (not context+" "+cand) -- tokenizing the candidate in isolation
    # can differ from tokenizing it after the context (leading-space/BPE-merge
    # artifacts). demo() checks this assumption holds for the tokenizer under test;
    # if a real model's tokenizer disagrees, this needs the same context-prefixed
    # tokenization the old method uses and a prompt_len diff instead.

    prefix_len = len(ctx_ids)
    n = len(candidates)
    pad_id = tokenizer.pad_token_id

    # step 1: forward the context once, keep the cache + the last-position logit
    # (predicts each candidate's first token)
    ctx_input = torch.tensor([ctx_ids], device=device)
    out1 = model(input_ids=ctx_input, use_cache=True)
    past = _expand_cache(out1.past_key_values, n)
    first_tok_logprob_row = torch.log_softmax(out1.logits[0, -1].float(), dim=-1)  # (vocab,)

    # step 2: batch the candidates' continuations (token 1.. of each candidate),
    # right-padded, forwarded against the shared expanded cache
    cand_lens = [len(ids) for ids in cand_id_lists]
    max_len = max(cand_lens)
    cont_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    cont_mask = torch.zeros((n, max_len), dtype=torch.long)
    for i, ids in enumerate(cand_id_lists):
        cont_ids[i, :len(ids)] = torch.tensor(ids)
        cont_mask[i, :len(ids)] = 1
    cont_ids, cont_mask = cont_ids.to(device), cont_mask.to(device)

    full_attn_mask = torch.cat([torch.ones(n, prefix_len, dtype=torch.long, device=device), cont_mask], dim=1)
    position_ids = torch.arange(prefix_len, prefix_len + max_len, device=device).unsqueeze(0).expand(n, -1)

    if max_len > 1:
        out2 = model(input_ids=cont_ids, attention_mask=full_attn_mask, past_key_values=past,
                      position_ids=position_ids, use_cache=False)
        cont_logprobs = torch.log_softmax(out2.logits[:, :-1], dim=-1)  # (n, max_len-1, vocab)
    else:
        cont_logprobs = None

    results = []
    for i, ids in enumerate(cand_id_lists):
        lp = first_tok_logprob_row[ids[0]].float().item()
        for t in range(1, len(ids)):
            lp += cont_logprobs[i, t - 1, ids[t]].float().item()
        results.append(lp)
    return results


@torch.inference_mode()
def score_candidates_full_reprocess(model, tokenizer, context, candidates, device):
    """Old method, verbatim shape from score_candidates.py / the notebook's
    score_candidates_batch -- reference for the demo's numerical comparison."""
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
        input_ids[i, :len(s)] = torch.tensor(s)
        attn[i, :len(s)] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)
    logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
    logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = input_ids[:, 1:]
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    start = prompt_len - 1
    return [token_lp[i, start:start + clen].sum().item() for i, clen in enumerate(cand_lens)]


def demo():
    """CPU correctness check, no Kaggle/local-GPU spend: a real tiny causal LM
    (distilgpt2), same Cache API as Qwen/Mistral, confirms the prefix-cache method's
    log-probs match the old full-reprocess method's -- the thing that would silently
    corrupt every downstream score if the cache-expansion or position_ids math were
    wrong."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.eval()

    context = "The quick brown fox jumps over the lazy"
    candidates = ["dog", "cat", "dogs", "doghouse"]  # mixed 1/2-token candidates

    # tokenization-boundary assumption check (see score_candidates_prefix_cache's note)
    for cand in candidates:
        a = tok(" " + cand, add_special_tokens=False)["input_ids"]
        b = tok(context + " " + cand, add_special_tokens=False)["input_ids"][len(tok(context, add_special_tokens=False)["input_ids"]):]
        assert a == b, f"tokenization boundary mismatch for {cand!r}: {a} vs {b}"

    old = score_candidates_full_reprocess(model, tok, context, candidates, "cpu")
    new = score_candidates_prefix_cache(model, tok, context, candidates, "cpu")

    for cand, o, n in zip(candidates, old, new):
        assert abs(o - n) < 1e-3, f"{cand!r}: old={o:.6f} new={n:.6f} diff={abs(o-n):.6f}"
        print(f"{cand:12s} old={o:.4f}  new={n:.4f}  diff={abs(o-n):.2e}")
    print("demo ok -- prefix-cache scoring matches full-reprocess scoring")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        demo()
