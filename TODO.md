# TODO — transformer fine-tune (new_data branch)

## Task

Predictive keyboard: given left context (space-tokenized string) + the first letter of
the next word, predict the word. Only metric that matters: **word-category accuracy**
(symbol/number are near-ceiling already, see `report.md`). Full dataset spec, EDA,
n-gram baseline (55.51% word acc, 60.12% overall) in `report.md`.

## Computer spec

- GPU: NVIDIA RTX 4060 Laptop, **8GB VRAM**
- CPU: AMD Ryzen 7 7435HS, 16 threads
- RAM: 29GB total
- Disk: 655GB free
- venv already has: torch 2.11+cu128, transformers 5.5.0, peft 0.20.0,
  bitsandbytes 0.50.2, accelerate 1.14.0, unsloth 2026.9.2

## Data statistics (measured, not estimated)

| | n | min | p10 | p25 | median | mean | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| train sentence length (tokens) | 3,803,957 | 10 | 23 | 28 | 33 | 33.29 | 38 | 44 | 57 | 135 |
| dev context length (tokens) | 94,488 | 7 | 17 | 20 | 24 | 24.57 | 29 | 33 | 42 | 71 |
| dev answer length (chars) | 94,488 | 1 | 1 | 2 | 4 | 4.43 | 6 | 8 | 12 | 18 |

Thin long tail (p99 under 60 tokens everywhere) — a training example (context + `[letter]`
+ target word, tokenized) will almost never need more than ~60-70 tokens. Confirms
optimization #1 below: `max_len=256` in `train_transformer.py` is ~4x more than needed,
pure padding waste.

`first letter` in dev: 47 distinct values, skewed — `t`/`a`/`s` alone are ~30% of rows
(12,934 / 9,921 / 6,751 of 94,488). Not uniform over the alphabet; matches English
word-initial letter frequency.

## Model chosen

~~mistralai/Mistral-7B-v0.1~~ → **Qwen/Qwen2.5-3B** (switched, see Current approach below).
Original pick reasoning logged at end of `report.md`: HellaSwag 84.0%/83.2%
(self-report + independent reproduction, cross-checked against ARC/MMLU). Both open
weight (Qwen2.5-3B: Apache 2.0 for 3B size). Zero-shot measurement on the actual task
(word-category accuracy on dev, not HellaSwag) showed Qwen2.5-3B matches/beats
Mistral-7B (70.26% vs 69.92% word acc) at ~1.7x the throughput, so kept the smaller
model — real task measurement overrides the benchmark-based estimate.

## Objective

Achieve 80% accuracy, via QLoRA
fine-tune instead of full fine-tune (won't fit 8GB VRAM otherwise).

## Current approach

- `scripts/train_transformer.py` — QLoRA (peft + bitsandbytes NF4), builds training
  examples on the fly from `train_final.src.tok`: random split point per sentence,
  `context + " [" + letter + "]"` -> target word, loss masked to target+EOS only.
  Self-check passing (`--demo` flag).
- Not yet run: model load + training deferred earlier for RAM headroom, now clear.
- `scripts/infer_transformer.py` — **written, running** (Suggestion step 1: measure
  ceiling before training). Zero-shot base Mistral-7B-v0.1, no fine-tune, no LoRA.
  Symbols separated out per this session's request: model only queried when the given
  first letter is alphanumeric; non-alnum first letters predicted trivially as the
  letter itself (same rule as `infer_ngram.py`). Letter constraint applied at the
  tokenizer's first-subword level (SentencePiece `▁`-prefix), not string-matched —
  correctness fix from Suggestion item 3. Greedy continuation batched round-by-round
  until a new word-boundary piece or EOS.
  - 64-row smoke test: 67.19% overall (word 66.67%, symbol 100%, n=5, number 0%, n=2 —
    too few number rows to mean anything). 320-row test: 74.06% overall, word 72.85%.
    Hand-checked predictions against the CSV — correct (`day`, `the`, `morning`, `was`,
    `to`, `a`, `that`, `of`, `women`), one known artifact (`moscow,` — trailing comma
    swept in because that piece isn't `▁`-prefixed in Mistral's tokenizer, doesn't hit
    the boundary-stop; scored wrong correctly, not fixed, rare edge).
  - Measured throughput: ~5.4 rows/sec, flat across batch size 16→64 — bottleneck is
    the no-KV-cache round loop (up to 5 full forward passes per batch, recomputes
    attention over the whole growing sequence every round instead of caching), not
    batch parallelism or compute. Full alnum dev (~85k rows) would be ~4.4h — killed
    that background run, too long, not worth it for a one-time measurement.
  - **Real number: random 2000-row sample (seed 42, `--limit 2000 --seed 42`), 378s
    (~6.3 min).** word 69.92% (1232/1762), symbol 99.48% (193/194, at EDA ceiling),
    number 0.00% (0/44 — model never produces the anonymized digit-placeholder format,
    structural, not fixable by more data), overall 71.25%.
  - **Zero-shot ceiling: word accuracy ~70%.** Already clears n-gram's 55.51% with zero
    training. 80% target is not in reach zero-shot — fine-tuning needs to close a real
    ~10pt gap, not a rounding error.
  - Output: `weights/dev_predictions_zeroshot_sample2k.csv` (2000-row sample; the killed
    full-dev file `weights/dev_predictions_zeroshot.csv` doesn't exist/is incomplete).

- **KV-cache rewrite done.** `predict_batch` now uses `model.generate()` (native cache)
  instead of the manual full-recompute-per-round loop — `FirstTokenLetterMask`
  (`LogitsProcessor`) constrains only the forced first piece, `StopAtWordBoundary`
  (`StoppingCriteria`) halts each row once a *later* generated token is boundary-tagged
  or EOS (must stay inert at the first step, since the forced first token is always
  boundary-tagged by construction — passing boundary ids as plain `eos_token_id`
  instead breaks this, halts every row after step 0, silently truncates every
  multi-piece word; hit this bug, fixed it).
  - Re-ran same 2000-row seed-42 sample: **69.92% word / 71.25% overall — exact match**
    to the pre-rewrite numbers, confirms correctness preserved.
  - **22.34 rows/sec vs 4.71 before — 4.7x faster.** Full dev (~85k alnum rows) would now
    be ~64 min instead of ~4.4h. Not run yet (only small-set requested so far).

- **`train_transformer.py` rewritten per Suggestion item 2.** Dropped the per-sentence
  `context + "[letter]" -> target` objective entirely. Now plain causal-LM fine-tune on
  raw lines from `train_final.src.tok` (every position supervised, ~30x more signal per
  token processed) via Unsloth `FastLanguageModel` + TRL `SFTTrainer(packing=True)`
  (packs many short lines into one training sequence, no per-line padding waste). The
  first-letter constraint moves entirely to inference (already correct in
  `infer_transformer.py`'s logit mask) — training never sees `[letter]` now, so nothing
  to keep consistent between train/infer on that front. `--demo` self-check passes
  (line loading + empty-line filtering). Not yet run against the real model.
- **Model-size check (Suggestion item 4) done — switched to Qwen2.5-3B.** Zero-shot
  Qwen2.5-3B on the same 2000-row seed-42 sample used for Mistral-7B: **word 70.26%,
  overall 71.55%** (vs Mistral-7B's 69.92%/71.25%) — 3B *wins*, not just "close enough."
  Per the Suggestion's own rule (take the 3B if 7B's lead is under ~3pt), switched
  `MODEL_ID` in `train_transformer.py` to `Qwen/Qwen2.5-3B`. Also ~1.7x faster zero-shot
  (38.14 rows/s vs 22.34, i.e. ~47s vs ~64min-scale for 2000/full-dev respectively).
  Hit + fixed a bug along the way: `infer_transformer.py`'s letter mask was sized to
  `len(tokenizer)`, but Qwen2.5's output embedding is padded wider (151936 vs 151665)
  — masked_fill shape mismatch. Fixed by sizing the mask off
  `model.get_output_embeddings().weight.shape[0]` instead. Also added `--model` flag
  and auto-detection of the word-boundary marker (`▁` SentencePiece vs `Ġ` GPT2-BPE) so
  the same script works across model families.
  - **Extended the shortlist to 8 models total, same 2000-row sample — Qwen2.5-3B wins
    clean.** Full comparison:

    | model | word acc | overall |
    |---|---|---|
    | **Qwen2.5-3B** | **70.26%** | **71.55%** |
    | Mistral-7B-v0.1 | 69.92% | 71.25% |
    | granite-3.1-2b-base | 68.10% | 69.65% |
    | Qwen2.5-1.5B | 67.99% | 69.55% |
    | gemma-2-2b | 67.99% | 69.55% |
    | Falcon3-3B-Base | 61.92% | 64.20% |
    | SmolLM2-1.7B | 61.07% | 63.45% |
    | phi-2 | 59.08% | 62.15% |

    Llama-3.2-3B untested — gated, access request denied. phi-2 is the only model
    besides Mistral-7B to score nonzero on `number` (9/44, 20.45%); everything else
    (including the winner) is flat 0% there — structural gap, not model-size-driven.

- **Root-caused + fixed the number-category 0% (was flagged "structural, not fixable"
  — it was, just not by the model).** Every `number` answer in dev is anonymized: the
  digit `1` repeated to the original number's length (`1234` -> `1111`), 100% of 2023
  dev number-rows confirmed this way, and `first letter` is always `'1'`. No LLM ever
  generates a repeated-`1` string (not real text) -- that's the actual reason every
  zero-shot model scored ~0% here, not a capability gap. A real word never starts with
  a digit, so a digit first-letter is an unambiguous number-placeholder signal --
  routed those rows around the model entirely (same trick as `symbol`), predicting the
  majority-class length ("11", 39.6% of dev number-rows are 2-digit; full distribution
  1-digit 30.7% / 2-digit 39.6% / 3-digit 18.4% / 4-digit 11.3%). Re-ran Qwen2.5-3B:
  number 0% -> **29.55%** (13/44 on the 2000-sample, sampling noise vs. the 39.6%
  dev-wide rate), overall 71.55% -> **72.15%**. Ceiling note: n-gram baseline already
  gets 71.68% on `number` by conditioning length on context -- this fix is the free
  flat win, not that; the smarter version is the ensemble opportunity in Suggestion
  item 5, not implemented.

## Known unresolved

- ~~Train scope: full 14M-example pass estimated ~3-6 days~~ — **stale, was for the old
  per-sentence objective.** With the plain-LM + packing rewrite, measured throughput on
  this 4060 plateaus at **~1150-1200 tok/s** (batch-size sweep: bs2 863, bs4 1057, bs8
  1142, bs16 1182 tok/s, VRAM-capped 6.9/8GB at bs16 — diminishing returns past bs8).
  Suggestion's 20-30M in-domain-token target now takes **~4.6-6.9h locally**, not days.
  Kaggle T4x2 should beat this (more VRAM headroom, bf16 LoRA without 4-bit quant) but
  local is now plausible too — decide after model-comparison zero-shot checks settle.
- Full-dev zero-shot run abandoned (~4.4h, too long) — 2000-row random sample (word
  69.92%, see Current approach above) stands as the ceiling estimate instead. ~2%
  margin of error at this n; good enough to decide the plan, not a final report number.
- Inference speed itself (no-KV-cache round loop) needs fixing before any post-training
  eval on 100k test lines, or that run is ~4h too.

## Optimizations identified, not yet applied

1. Cap `max_len` down from 256 to ~64 + drop outlier-long lines (measured: p99 under
   60 tokens across train/dev/test, see Data statistics above — current 256 cap wastes
   most compute on padding).
2. Filter training examples to word-category targets only (symbol/number are
   near-ceiling per EDA already, wasted training signal).
3. `bnb_4bit_use_double_quant=True` + gradient checkpointing → more VRAM headroom →
   bigger micro-batch → more throughput.
4. Length-bucketed batching (sort by token length before batching).
5. `torch.compile` — try after 1-4, not before.
6. Swap `AutoModelForCausalLM`/`peft` loader in `train_transformer.py` for Unsloth's
   `FastLanguageModel` (installed, unused so far) — claimed 2-5x + ~50% less VRAM.

## Target

Word accuracy > 55.51% (n-gram baseline) on `dev_set_final.csv`, measured the same way
(per-category breakdown: word / symbol / number), within a training budget that fits
an 8GB laptop GPU in well under a day.

---------------------------------------

# Suggestion

the biggest wins are not in the six listed optimizations, they're in the training formulation and in measuring before training. In priority order:

1. Measure the ceiling before you spend a GPU-day. Write infer_transformer.py first and run the base Mistral-7B, zero-shot, with the first-letter mask, on dev. That single ~30 min run tells you (a) where a 7B model starts (my estimate: 55–65% word accuracy zero-shot, assuming dev text looks like natural English), and (b) whether 80% is even in range. Also log top-5 constrained accuracy: if top-5 is ~85% and top-1 is ~62%, fine-tuning can close some of that gap; if top-5 is 75%, no amount of fine-tuning gets you to 80% and you should say so in the report rather than chase it. Given n-gram at 55.5% and a 7B pretrained model, my honest estimate is a fine-tuned ceiling around 68–75% on this task — 80% is likely above the intrinsic entropy of next-word prediction even with a first-letter hint. The dev run settles it.

2. Change the training objective — this is the 30x win. Your current setup builds one example per sentence (context + [letter] → one target word), so each ~60-token forward pass supervises one word. Instead, train as a plain causal LM on full sentences: every position is a target, so a 33-token sentence gives ~32 supervised predictions per pass. The first-letter conditioning belongs only at inference: P(w | ctx, letter) = P(w | ctx) / Σ_{w' starts with letter} P(w' | ctx) — exactly what the logit mask computes. The [letter] tag in training adds no information the mask doesn't already provide, and it costs you ~30x sample efficiency. This also dissolves optimization #2 (word-only filtering); if you want, just zero the loss on symbol/number tokens.

With this, "500k–1.5M examples" becomes "how many in-domain tokens can I push through": at a realistic ~1.5–2.5k tok/s for QLoRA-7B with Unsloth on a 4060 (measure it — pilot 200 steps), 20–30M tokens is 3–5 hours. That's 15–20% of your 130M-token corpus, far more supervision than the current plan.

3. Fix the tokenization boundary — this is probably why GPT-2 fine-tuned scored 36%, below the n-gram. Your data is space-tokenized (don 't, , as separate tokens?). Mistral/GPT-2 were pretrained on natural text; feeding them detokenized-looking strings with odd spacing hurts, and the letter mask must be applied to the first subword of the next word, i.e. tokens of the form ▁t…, case-insensitively if dev is lowercased. Then continue greedy until the next ▁-prefixed token or EOS and compare the assembled word. If you mask over all tokens starting with t (including mid-word pieces) or ignore the ▁, the decode is wrong. Check this against a few dev rows by hand before trusting any number. Whatever preprocessing you settle on, apply identically in training and inference.

4. Reconsider 7B. For a fixed-domain, top-1 next-word task, in-domain token volume matters more than base model size. A 1.5–3B model (Qwen2.5-1.5B/3B, Llama-3.2-3B) in bf16 LoRA — no 4-bit quant, no double-quant, ~3–4x the throughput — trained on 3–4x more of your corpus in the same wall-clock will plausibly match or beat QLoRA-7B on a subsample, and inference on 100k lines drops from ~30 min to ~10. Run the zero-shot check from step 1 on both a 3B and the 7B; if the 7B's zero-shot lead is under ~3 points, take the 3B.

5. Ensemble with the n-gram at the end. Interpolate λ·log P_lm(w) + (1−λ)·log P_ngram(w) over the candidate set (tune λ on a dev split). The n-gram memorizes domain-specific collocations the LM won't see in a partial epoch; this is usually worth 1–3 points for free.

On your six optimizations: #1 (max_len 64) and #3 (grad checkpointing) do them; #4 (length bucketing) matters less once you train on full sentences with packing; #5 skip; #6 (Unsloth) yes, it also handles packing for you. #2 becomes moot per step 2.

Suggested sequence: infer script → zero-shot 7B and 3B on dev → pick model → 200-step timed pilot → 3–5 h plain-LM run → ensemble → report the measured ceiling honestly alongside whatever number you hit.