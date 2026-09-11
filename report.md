# Report: n-gram baseline (new_data branch)

## Dataset spec

**Task:** predictive keyboard — given left context + the first letter of the next
word, predict the word.

**Files** (`data/`):

| file | rows/lines | notes |
|---|---|---|
| `train_final.zip` (member `train_final.src.tok`) | 3,803,957 lines | 126,652,466 tokens, 99,018 distinct tokens |
| `dev_set_final.csv` | 94,488 rows | has `answer` |
| `test_set_no_answer_final.csv` | — | no `answer` column |
| `gigaword.tar.gz` | 149 monthly files, 1994-07..2006-12 | raw NYT Gigaword, 2.4GB compressed; not used in train/dev/test, separate pretraining corpus |

**CSV schema** (dev/test): `context` (space-tokenized left context string), `first
letter` (single char constraining the answer), `answer` (target word, dev/train only).

**Train corpus** (`train_final.src.tok`): one sentence per line, pre-tokenized —
lowercased, punctuation isolated as its own token (`"o.j."` -> `o . j .`, `"don't"` ->
`don ' t`, `"well-known"` -> `well - known`). No PTB placeholders (`-lrb-`, `-rsb-`,
etc.) anywhere in this dataset.

**Answer categories** (regex: digits -> `number`, `[a-z']+` -> `word`, else `symbol`):
- `word` — ordinary vocabulary, open-class.
- `number` — every numeric value is anonymized to a digit-length placeholder (`1`,
  `11`, `111`, ... up to 8 repeated `1`s), never a real number. No OOV risk, but the
  placeholder discards the actual digits.
- `symbol` — punctuation. Near-fully deterministic from `first letter` alone (~99.6%
  majority-vote ceiling in dev).

**Known mismatch:** train's tokenizer folds every literal `[`/`]` into `[UNK]`
(414,394 occurrences), so train never emits a literal bracket. Dev/test still contain
literal `[`/`]` (dev: 124 bracket answers, 0.13% of rows; test: 47 raw `[` context
tokens, 353/60 rows with `[`/`]` as the given first letter) — a structural,
un-prunable train/eval gap, not a rare-word coverage issue.

**Length distribution** (tokens = whitespace-split, `data/*` measured directly):

| | n | min | p10 | p25 | median | mean | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| train sentence length | 3,803,957 | 10 | 23 | 28 | 33 | 33.29 | 38 | 44 | 57 | 135 |
| dev context length | 94,488 | 7 | 17 | 20 | 24 | 24.57 | 29 | 33 | 42 | 71 |
| test context length | 94,826 | 7 | 17 | 20 | 24 | 24.52 | 28 | 33 | 42 | 72 |
| dev answer length (chars) | 94,488 | 1 | 1 | 2 | 4 | 4.43 | 6 | 8 | 12 | 18 |

Train sentences are tighter and shorter than dev/test context length alone (median 33 vs
24) — expected, since dev/test `context` is a left-truncated prefix of a longer original
sentence, not a full sentence. Long tail is thin either way (p99 under 60 tokens on all
three); a handful of train outliers reach 135 tokens.

Dev's `first letter` covers 47 distinct values (26 letters + digits/punctuation), heavily
skewed: `t`/`a`/`s` alone cover ~30% of rows (12,934 / 9,921 / 6,751), matching English
letter-frequency-as-word-starts, not uniform over the alphabet.

Full EDA lives in `eda/*.ipynb`.

## Key EDA findings

- **Vocab coverage is tail-heavy.** 90% of train's token mass is covered by ~4,400 words
  (4.4% of the 99,018-word vocab); 99% needs ~30,800 words. Dev/test need a
  proportionally larger fraction of *their own* vocab for the same coverage (57% vs 31%
  for train) since they're much smaller corpora, not because the words differ.
- **Answer categories** (regex-based: digits -> `number`, `[a-z']+` -> `word`, else
  `symbol`) split dev as 83,228 word / 9,237 symbol / 2,023 number rows.
- **Symbols are ~99.6% deterministic from the first letter alone** (majority-vote
  ceiling; 24/27 first-letters fully deterministic in dev, only `1`, `e`, `u` ambiguous).
- **Numbers are digit-length placeholders, not real values** (`1`, `11`, `111`, ...
  up to 8 repeated `1`s). No OOV risk, but first-letter alone is a weak predictor
  (39.6% majority-vote accuracy in dev) since every number placeholder starts with `1`.
- **Bracket tokenization mismatch.** Train's tokenizer folds every `[`/`]` into
  `[UNK]` (414,394 occurrences), never emitting a literal bracket. Dev/test keep
  literal `[`/`]` in places: dev has 124 rows (0.13%) with literal bracket answers,
  test has 47 raw `[` context tokens and 353/60 rows with `[`/`]` as the given first
  letter. In dev, first-letter `[` -> true answer is literally `[` 63/63 times (ratio
  1.0), never `[UNK]` — so the model structurally cannot get these right; it's a fixed
  ~0.13% ceiling loss, not a vocab-size problem. No PTB-style placeholders
  (`-lrb-`, `-rsb-`, etc.) exist anywhere in this dataset.
- **Rare-word cutoff vs. dev OOV** — dropping the rarest N% of train vocab costs very
  little dev-answer OOV, since dev is skewed toward common words:

  | vocab dropped | min unigram count kept | train freq mass kept | dev OOV |
  |---|---|---|---|
  | 0% | - | 100.00% | 0.13% (baseline) |
  | 20% | >8 | 99.91% | 0.19% |
  | 30% | >10 | 99.84% | 0.25% |
  | 40% | >15 | 99.74% | 0.33% |
  | 50% | >23 | 99.59% | 0.47% |
  | 70% | >74 | 98.94% | 1.06% |

## N-gram model

`scripts/train_ngram.py` — custom pure-Python count-table trainer (not kenlm's
`lmplz`; `scripts/build_kenlm.sh` only compiles that binary, unused so far). Trains
order-1..n count tables, `<s>`/`</s>` padded, encoded to int ids.

**Order chosen: n=4.** Dev context length is never the bottleneck (min 7 tokens,
median 24 — plenty of room even for n=5). Went with 4 over 5 because sparsity is
already severe at order 4 (see fallout below); a 5-gram would push almost every
context into the "fully sparse" regime with no real gain, more RAM/time for it.

**Pruning: `--min-count 10`**, applied to `(context, word)` pair counts at order ≥2
only (order-1 unigrams always kept intact as final backoff). Note this is a
different mechanism from the EDA's vocab-level word cutoff above — it prunes rare
*co-occurrences* per context, not whole words out of the vocabulary, and a context is
never fully emptied: if every continuation of a context has count <10, the whole
unpruned set is kept for that context instead of being deleted (`or d` fallback in
`train.py`) so backoff never bottoms out at nothing.

Fallout, measured on the saved model (`weights/ngram_4_a.bin`, 684MB):

| order | contexts | surviving entries | contexts hit "keep intact" fallback (all-rare) |
|---|---|---|---|
| 1 (unigram) | 1 | 99,019 | n/a (never pruned) |
| 2 (bigram) | 99,019 | 1,260,119 | 54.8% |
| 3 (trigram) | 6,384,545 | 14,942,936 | 94.1% |
| 4 (4-gram) | 27,543,041 | 43,771,038 | 98.0% |

Pruning barely touches order-3/4: the data's already sparse enough there that nearly
all contexts have only rare continuations and pass through untouched. Real pruning
happens mostly at order-2 (bigrams), where common contexts have a long rare-word
tail worth trimming. This is also why a 5-gram is unlikely to help — order-4 is
already 98% "everything is rare," order-5 would be worse with no headroom for
pruning to even matter.

Training: 3,803,957 lines, ~13 minutes.

## Inference & accuracy (dev set) — model A

`scripts/infer_ngram.py` — backs off order 4 -> 3 -> 2 -> 1, filtered to candidates
starting with the given first letter, argmax count. Symbol category is predicted
trivially as the given first letter itself (EDA already showed this is ~99.6%
optimal; no need to route it through the n-gram). Predictions saved to
`weights/dev_predictions_a.csv` (context, first letter, answer, category, prediction,
correct).

| category | correct | total | accuracy |
|---|---|---|---|
| word (alpha) | 46,197 | 83,228 | **55.51%** |
| symbol | 9,157 | 9,237 | **99.13%** |
| number | 1,450 | 2,023 | 71.68% |
| overall | 56,804 | 94,488 | 60.12% |

Symbol accuracy is at the EDA ceiling — nothing to gain there. Word accuracy (55.5%)
is the number that matters for judging this pipeline; number accuracy is a bonus, not
requested but included for completeness.

## Model B: same 4-gram, trained on gigaword instead of train

Idea: mask gigaword onto train's exact vocab (any word not in train's 99,018-word
vocab -> `[UNK]`, same tokenization rules as `tokenize_gigaword.py` already had),
then train a 4-gram on that much bigger corpus, see if more data beats train's
narrower-but-matched distribution.

**Masking** (`scripts/tokenize_gigaword.py`, defaults repointed at
`train_final.src.tok` / output `data/gigaword_masked.tok`): streamed straight from
`gigaword.tar.gz` (149 monthly files), never extracted to disk. 31,320,304 lines,
1,399,680,893 tokens, 74,055,528 `[UNK]` (5.29%). Output file 6.7GB. 17m47s, RSS
stayed under 300MB the whole time — this step was never the memory risk.

**Training hit a wall the first time.** `train_ngram.py`'s `--min-count` was a
post-pass filter: it counts every `(context, word)` pair for the *entire* corpus
first, into plain nested dicts, and only prunes after that full pass finishes. That
peak (all distinct n-gram windows before any filtering) is what blows memory, not
the final pruned size — so `--min-count` didn't help at all. First attempt, capped
at 14GB (`ulimit -v`), died with `MemoryError` at 13m56s, still inside the counting
loop, before pruning ever ran.

**Fix: prune during counting, not after.** Patched `train_ngram.py` with
`--prune-every-lines N` — every N lines, compact all order ≥2 count tables in place
(drop `(ctx, word)` pairs below `--min-count`; unlike the final pass, a context that
ends up fully empty is just deleted, no fallback-to-unpruned). This is approximate
(a pair's count can be undercounted if its occurrences are spread across prune
checkpoints — it gets evicted, then starts recounting from 0 if it recurs) but
that's an acceptable trade for staying inside a fixed RAM budget, which a one-shot
end filter fundamentally can't do.

Trained with `--n 4 --min-count 10 --prune-every-lines 1000000` (≈31 checkpoints
over the 31.3M lines), still under the 14GB cap throughout — RSS held around 7-8GB
for the whole run instead of climbing unbounded. 31,320,304 lines, 155m31s (2h35m).
Saved to `weights/ngram_4_b.bin`, 197MB (smaller than model A's 684MB despite 11x
more input data — checkpointed pruning evicts rare entries continuously instead of
once, so far less survives to the final save).

| order | contexts | surviving entries | fallback ctx (all-rare, kept intact) |
|---|---|---|---|
| 1 (unigram) | 1 | 93,278 | n/a |
| 2 (bigram) | 71,853 | 1,494,565 | 11.3% |
| 3 (trigram) | 2,385,340 | 4,829,833 | 63.4% |
| 4 (4-gram) | 7,352,980 | 9,365,827 | 76.2% |

Model B's own vocab is 93,279 (< train's 99,018) — some train words simply never
occur in gigaword even once. Compared to model A, every order has far fewer
surviving contexts (order-4: 7.35M vs 27.5M) despite 11x the training data — the
periodic hard pruning is a much blunter instrument than the one-shot end filter.

### Inference & accuracy (dev set) — model B

Same `infer_ngram.py`, `--model ngram_4_b.bin`. Predictions saved to
`weights/dev_predictions_b.csv` (same schema as model A's).

| category | correct | total | accuracy | vs. model A |
|---|---|---|---|---|
| word (alpha) | 37,592 | 83,228 | 45.17% | -10.34pp |
| symbol | 9,157 | 9,237 | 99.13% | +0.00pp (identical rule) |
| number | 709 | 2,023 | 35.05% | -36.63pp |
| overall | 47,458 | 94,488 | 50.23% | -9.89pp |

**Model B is worse across every category that isn't the fixed symbol rule.** More
data did not beat train's narrower, better-matched distribution here. Three likely
compounding causes: (1) gigaword is off-domain relative to train/dev even after
vocab masking — same NYT-family source but presumably a different cleaning/split
than what train/dev were built from; (2) the streaming prune is lossier than the
one-shot filter (undercounts split across checkpoints), directly shrinking exactly
the long tail that word-level accuracy depends on; (3) number accuracy tanking hardest
(35% vs 72%) fits both explanations — gigaword's digit-placeholder distribution by
length likely differs from train/dev's, and that signal is exactly the kind of
low-count-per-context detail streaming pruning is roughest on.

**Verdict: keep model A, don't ship model B as-is.** If gigaword is worth
revisiting, the fix isn't more pruning — it's exact counting at this scale (i.e.
actually use the already-built `scripts/bin/lmplz`/kenlm, which does external-memory
exact counting, instead of the approximate in-RAM streaming prune here).

## Open items

- kenlm's `lmplz`/`build_binary` are built (`scripts/bin/`) but unused — current model
  is the custom Python trainer, not a true kenlm ARPA/binary LM.

## Next: pretrained transformer base model

Initial pick by benchmark (HellaSwag, cross-validated against ARC/MMLU) was
**mistralai/Mistral-7B-v0.1**. That was a proxy — HellaSwag isn't this task (no
first-letter constraint, multiple-choice not open-vocab). Actually measuring the real
task (zero-shot, no fine-tune, on `dev_set_final.csv`) overturned it.

**Final pick: [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)** — open
weight, Apache 2.0, no gated access. Beats Mistral-7B on the actual task despite being
less than half the size, and ~1.7x faster at inference. Real-task measurement beats a
benchmark proxy every time it's cheap enough to run — it was here (~1 min per model
on a 2000-row sample).

**Model shortlist, zero-shot (no fine-tune), same random 2000-row seed-42 sample of
`dev_set_final.csv`, first-letter-constrained greedy decoding:**

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

Llama-3.2-3B untested (gated, access request denied). Every model here scored ~0% on
`number` in this table except phi-2 (20.45%) — see below, that's not a model gap.

**Zero-shot ceiling for Qwen2.5-3B: word 70.20%, already well clear of the n-gram
baseline's 55.51% with zero training** — fine-tuning needs to close a real gap, not a
rounding error, and a flat 80% target is likely above this task's intrinsic entropy
(see `TODO.md`'s pasted "Suggestion" for the reasoning).

**Number-category finding:** every `number` answer in dev is anonymized — the digit
`1` repeated to the original number's length (`1234` → `1111`; confirmed on all 2023
number-rows in dev, 100%), and `first letter` is always `'1'`. No LLM generates a
repeated-`1` string (not real text), which is the actual reason every model above
scores near-0% there — not a capability gap. A real word never starts with a digit,
so a digit first-letter is an unambiguous number-placeholder signal, routed around the
model the same way `symbol` already is. The n-gram baseline already exploits this
(71.68% on `number`, by conditioning the digit-length on context) — better than a flat
majority-length guess (39.6%, the most common length in dev).

**Full production config for Qwen2.5-3B — word from the LLM, number from the n-gram,
symbol deterministic (letter copy), same 2000-row sample:**

| Alpha (LLM, word) | Number (n-gram) | Symbol (deterministic) | **Total** |
|---|---|---|---|
| 70.20% (1237/1762) | 81.82% (36/44) | 99.48% (193/194) | **73.30% (1466/2000)** |

This ensemble (`scripts/infer_transformer.py --ngram-model ...`) is the real
zero-shot floor to beat post-fine-tune, not the flat-guess 72.15% or the pure-LLM
71.55% numbers above.

**Checked whether Qwen3.5 (released March 2026, after the original shortlist) beats
Qwen2.5-3B — it doesn't.** Same 2000-row seed-42 sample, `--mask-number` mode: number
rows also go through the model (not the n-gram/flat-guess routing above), predicted
digit string masked to `1`×length before scoring — a length-only check, no ensemble:

| model | word acc | number (masked, length-only) | overall | rows/s |
|---|---|---|---|---|
| **Qwen2.5-3B** (kept) | **70.26%** | — (not run this way) | 71.55% | 38.14 |
| Qwen3.5-4B-Base | 66.46% | 11.36% | 68.45% | 9.13 |
| Qwen3.5-2B-Base | 61.18% | 25.00% | 64.10% | 15.70 |

Qwen2.5-3B still wins on word accuracy and is 2.5-4x faster per row in this venv.
Qwen3.5 is a hybrid linear-attention/Mamba-style architecture (not plain transformer
attention like Qwen2.5) tuned for long-context/multimodal use, not short next-word
completion — and this venv lacks the `flash-linear-attention`/`causal-conv1d` kernels
its fast path needs (ABI mismatch: prebuilt `.so`s are cp310, venv is cp312), forcing a
slow torch fallback, though that's a speed penalty only, not a correctness one. Newer
and bigger did not mean better here — kept Qwen2.5-3B.

## Fine-tuned Qwen2.5-3B LoRA (Kaggle, T4, ~8h total across 2 sessions) — top-1/5/10 + error EDA

Trained via `scripts/train_transformer.py` / `kaggle/qwen3b_train_infer.ipynb` on
`train_final.src.tok` (plain packed-sequence causal-LM objective, no
`context+"[letter]"→target` prompt — first-letter constraint is entirely an inference-
time logit mask, see the script's docstring). Two sessions (`TRAIN_SEED` bumped between
them to avoid replaying the same epoch-0 shuffle), ~6h + ~2h wall-clock, LoRA
r=16/α=32 on q/k/v/o_proj. `scripts/infer_topk.py` evaluates top-1/5/10 via beam search
(`num_beams=num_return_sequences=k`, letter-masked at step 0, stopped at the next word
boundary) on a stratified 10% dev sample (9448 rows); top-k = top-k *distinct* words,
beam duplicates collapsed. Number category still uses the `--mask-number` length-only
check (candidates masked to `1`×length before scoring, no ensemble).

| | top1 | top5 | top10\* |
|---|---|---|---|
| overall | 71.45% (6751/9448) | 83.86% (7923/9448) | 83.86% (7923/9448) |
| word | 69.72% (5716/8198) | 84.02% (6888/8198) | 84.02% (6888/8198) |
| symbol | 98.76% (1035/1048) | 98.76% | 98.76% |
| number | 0.00% (0/202) | 0.00% | 0.00% |

\*top10 run with `k=5` (beam width), so the top10 column is identical to top5 here —
not a real top-10 ceiling, just what 5 beams cap out at. k=5 chosen over k=10 for
~1.8x the throughput (7.6 vs 4.05 rows/s) at effectively the same top5 number, since
duplicate-collapsed beams past ~5 rarely added a new distinct word anyway.

Number 0% even with 5 candidates is a placeholder-token limitation (see the
Number-category finding above), not something top-k or a reranker fixes — no LLM beam
naturally emits a repeated-`1` string.

**Error EDA on the 2482/8198 word rows still wrong at top1:**
- **100% of wrong top1 predictions start with the correct letter** — the mask is doing
  its job; every miss is a *within-letter* ranking error.
- **47% (1172/2482) are recovered by top5** — beam already had the right answer, just
  ranked below #1. The other 53% are true beam misses.
- **Dominant failure mode: function-word ambiguity**, genuinely unknowable from
  left-context alone — a↔an (29+24), that→to (14), and→at (11), their→the (11),
  the→that (10), for→from (7), on→of (7). These aren't model errors so much as valid
  alternative continuations; the task itself is ambiguous at these positions.
- **Register mismatch**: `u`→`us` (13x) — informal "u" (=you) in the corpus, model
  defaults to the formal completion.
- **Unknowable factual picks**: sunday→saturday (9), thursday→tuesday (6) — no signal
  in left context to disambiguate a specific weekday.
- **Plausible near-synonyms on content words**: visibility→visitors,
  initiatives→investments, resistance→rebels/remnants — right gist, wrong exact word.
- **By first letter** (word category, top1/top5%): worst are high-branching-factor
  consonant starts — r 58.0%/74.7%, u 58.0%/79.5%, g 59.7%/78.3%, s 60.2%/75.3%,
  c 60.8%/74.4% (many common function/content words compete). Best are low-branching
  or rare-letter starts — x 100%(n=3), y 91.2%/98.3%, o 86.4%/94.7%, z 85.7%(n=7),
  t 79.4%/90.3% (t/o dominated by "the/to"/"of/on", where the LM's prior is very
  strong despite huge n). Pattern tracks branching factor / function-word density at
  that letter, not the letter itself.

**MiniLM rerank of the top5 beam — tried, hurts accuracy.** Reranked each row's top5
candidates by cosine similarity between a `sentence-transformers/all-MiniLM-L6-v2`
embedding of the last-12-word context and an embedding of each candidate word (mean-
pooled, normalized), independent one-off script (not committed):

| | top1 acc |
|---|---|
| beam log-prob order (baseline) | 69.72% (5716/8198) |
| MiniLM-cosine reranked | 58.31% (4780/8198) |
| top5 ceiling | 84.02% (6888/8198) |

-11.4pt, net loss (439 flips wrong→right vs. 1375 right→wrong). MiniLM's generic
topical-similarity score is blind to grammar/agreement/tense/recency — it prefers
`deaths` over `death`, `an` over `a`, `because` over `by`, `fourth` over `first`,
exactly the syntactic signal the causal LM's own beam score already encodes for free.
Sentence-embedding cosine similarity is the wrong tool for next-token reranking; it's
built for retrieval/topical-similarity, not fluency/grammaticality.

**Trained cross-encoder reranker (MiniLM, listwise cross-entropy over the top5) —
also tried, also hurts.** Pilot (`scripts/train_reranker.py`, n=3000 rows generated
from a train-split chunk the LoRA fine-tune never saw): 69.72% → 62.34% top1. Better
than the frozen-cosine attempt but still net-worse than the beam's own ranking.
Per the pilot's own go/no-go rule, not scaled to the full run — a trained reranker
only ever sees the same left-context the LM already conditioned on, no new
information, so it's fighting for scraps of a signal the causal LM's beam score
already has, not adding a fresh one.

## Matched zero-shot vs fine-tuned comparison + n-gram blend

**Resolved whether the LoRA fine-tune (~8h/~25M tokens across 2 Kaggle sessions) was
worth it.** The earlier comparison (zero-shot 70.20-70.26% vs fine-tuned 69.72%) mixed
greedy decoding (zero-shot) against beam-top1 (fine-tuned) — not apples to apples.
Reran both through the identical beam-k5 protocol, same 8198 word rows
(`scripts/gen_matched_scores.py`, local 4060 — Kaggle's T4 measured slower here even
after fixing an accidental multi-GPU model shard, ~1.5 rows/s vs local's 7.5-8.75,
so this ran local instead):

| | top1 | top5 |
|---|---|---|
| zero-shot (beam k=5) | 66.52% (5453/8198) | 82.11% (6731/8198) |
| fine-tuned LoRA (beam k=5) | **69.72%** (5716/8198) | **84.02%** (6888/8198) |

**+3.2pt top1 under matched protocol — fine-tuning genuinely helped.** The earlier
flat-looking result was a decoding-method artifact (beam search apparently hurts the
zero-shot model's calibration more than greedy does), not evidence the training was
wasted.

**N-gram blend** (`scripts/blend_ngram.py`): `score(w) = λ·logP_lm(w) + (1-λ)·logP_ngram(w)`,
candidate set = fine-tuned LM's top5 ∪ the n-gram's own best guess (so the blend can
recover a word the LM's beam never generated, not just rerank its list).

First pass tuned λ on a single stratified 80/20 split (λ=0.90, 72.50% word on the
1640-row word holdout) — **that number turned out to be small-sample noise**, not a
real effect: n=1640 carries ±2pt at 95% CI. Replaced with pooled 5-fold stratified
CV (every one of the 9448 rows held out exactly once, tuned on the other 4 folds each
time) for a trustworthy estimate:

| | pooled 5-fold CV accuracy |
|---|---|
| word (blended) | **71.52% (5863/8198)** |
| symbol (deterministic) | 98.76% (1035/1048) |
| number (n-gram only) | 68.32% (138/202) |
| **overall** | **74.47% (7036/9448)** |

Beats pure n-gram (55.51% word), pure zero-shot LLM (66.52-70.2%), and pure
fine-tuned LLM (69.72%) — real margin (+1.8pt word over fine-tuned LLM alone), just
smaller than the noisy first read suggested.

**KenLM modified-Kneser-Ney 5-gram, swapped in as the blend's n-gram scorer**
(status.md advice item 1 — built `tools/lmplz` from source, trained on the full
130M-token `train_final.src.tok`, replaces the old unsmoothed relative-frequency
`score_word`). Compared under the *identical* pooled-CV harness to isolate the
effect of the scorer swap alone:

| n-gram scorer | word | overall |
|---|---|---|
| old (unsmoothed counts, backoff 4→1) | 71.52% (5863/8198) | 74.47% (7036/9448) |
| KN5 (KenLM, modified Kneser-Ney) | **71.74% (5881/8198)** | **74.66% (7054/9448)** |

**+0.22pt word, +0.19pt overall** — real (both measured on the same 9448 rows,
same folds) but well under the advice's +1-2pt estimate. Smoothing quality wasn't
the bottleneck here; the blend's ceiling looks set mostly by the candidate set
(LM top5 ∪ n-gram top1), not by how well either side's probabilities are calibrated.
Kept KN5 as the default scorer (strictly better, free).

**Direct candidate scoring** (status.md advice item 5, `scripts/score_candidates.py`):
instead of trusting `generate()`'s own beam sequence log-prob, score every candidate
in the union set (LM's beam top5 ∪ n-gram's own top10 for that letter — widened from
top1) with one teacher-forced batched forward pass, summing full-word log-prob.
Removes beam-search pruning distortion on multi-subword words and gives scores that
are directly comparable across candidates the beam itself never fully explored.
Compared under the identical pooled-CV harness, KN5 scorer both times:

| LM scoring | word | overall |
|---|---|---|
| beam sequence_scores (top5 only) | 71.74% (5881/8198) | 74.66% (7054/9448) |
| teacher-forced (top5 ∪ ngram-top10 union) | **72.62% (5953/8198)** | **75.42% (7126/9448)** |

**+0.88pt word, +0.76pt overall** — the largest single gain of this round, and
consistent with a 200-row pilot that showed the same effect (72.5% top1) before
committing to the full run. **New best pipeline result (going into item 2):
72.62% word / 75.42% overall** (beats fine-tuned LLM alone by +2.9pt word).

**Learned blender** (status.md advice item 2, `scripts/learned_blender.py`):
replaced the scalar λ with `sklearn.HistGradientBoostingClassifier` (already
installed, no new dependency), pointwise-to-listwise over 7 features per candidate —
teacher-forced LM log-prob, LM rank, KN5 log-prob, n-gram backoff level (highest
order the candidate was actually seen at — genuinely new info the LM's own
log-prob doesn't carry), unigram log-frequency, word length, is-function-word.
Same pooled 5-fold CV:

| blend method | word | overall |
|---|---|---|
| scalar λ (KN5 + teacher-forced) | **72.62% (5953/8198)** | **75.42% (7126/9448)** |
| learned GBDT blender | 72.55% (5948/8198) | 75.37% (7121/9448) |

**Flat, marginally worse — abandoned.** The extra features didn't add anything the
1-parameter λ blend wasn't already capturing: backoff level and unigram frequency
correlate heavily with KN5's own log-prob (that's literally what KN5 is built from),
and is-function-word/word-length carry too little independent signal at this
candidate-set size to earn their model complexity. Same root lesson as the MiniLM
reranker post-mortem above — more model complexity doesn't help when the "new"
features aren't actually independent of what's already blended.

**Second LLM in the blend** (status.md advice item 4, `scripts/score_candidates.py`
+ `scripts/blend3.py`): rather than fine-tuning a second model, added **Mistral-7B-v0.1
base, zero-shot**, teacher-forced-scored on the exact same candidate set already used
for Qwen (LM top5 ∪ n-gram top10 — no new candidate generation, no beam search for
Mistral at all, just one forward pass per row reusing `score_candidates.py`). Chosen
for genuine architectural diversity, not solo accuracy: different pretraining corpus,
different tokenizer (SentencePiece vs Qwen's byte-BPE) — checked two truly
different-architecture options first (state-space `mamba-2.8b`: 65.55% word zero-shot;
pure-RNN `RWKV-4-3b`: 60.66%) but both trailed Qwen by 4.7-9.6pt, too far behind to
expect a net blend gain; Mistral's -0.34pt solo gap was the safer bet.

3-way blend: `score(w) = λ_qwen·qwen_tf(w) + λ_mistral·mistral_tf(w) + (1-λ_qwen-λ_mistral)·kn5(w)`,
2D grid search over the (λ_qwen, λ_mistral) simplex, same pooled 5-fold CV, KN5 scores
precomputed once per row (not per grid point — a 2D grid recomputing KenLM inside the
inner loop would have taken ~37 hours; precomputing brought the whole grid search
under 2 minutes):

| blend | word | overall |
|---|---|---|
| 2-way (Qwen tf + KN5) | 72.62% (5953/8198) | 75.42% (7126/9448) |
| 3-way (+ Mistral tf) | **74.64% (6119/8198)** | **77.18% (7292/9448)** |

**+2.02pt word, +1.76pt overall — the largest single gain of the entire session**,
bigger than items 1, 2, and 5 combined. Tuned weights: λ_qwen≈0.20-0.30,
λ_mistral≈0.50-0.60 (KN5 gets the remainder). **Mistral is weighted MORE than Qwen**
despite scoring slightly lower solo zero-shot (69.92% vs 70.20-70.26%) — direct
evidence that the architecture-diversity bet was the right call: Mistral's errors are
decorrelated enough from Qwen's that the blend leans on it harder than raw solo
accuracy would predict. Result at this point: 74.64% word / 77.18% overall
(+4.9pt word over fine-tuned LLM alone). Kept as production (KN5 + teacher-forced +
Mistral 3-way blend).

**Number-only KN5** (`scripts/extract_gigaword_numbers.py`, `scripts/filter_train_numbers.py`,
`scripts/eval_number_kn5.py`): the general n-gram routes the number category alone
(word/symbol don't touch it), but it's trained on the same corpus as everything else —
no new information for numbers specifically. Built a second KN5, this one trained ONLY
on number-containing lines, drawing on Gigaword (previously unused — training on it in
full is too slow, but a digit-only slice is tiny) plus `train_final.src.tok`'s own
number lines:

1. Stream `gigaword.tar.gz`, keep only paragraphs with ≥1 digit token, anonymize each
   digit run to `"1"*len` (matching `train_final.src.tok`'s own scheme — confirmed only
   lengths 1-8 ever appear there) *before* the vocab lookup, not after — a literal `"37"`
   was never going to be in-vocab, `"11"` already is. 10.28M/31.3M paragraphs kept.
2. Filter `train_final.src.tok` to its own number-containing lines the same way (no
   anonymization needed, already done) — 1.27M/3.80M lines kept.
3. Concatenate (11.56M lines, 606M tokens) and train a 5th-order KenLM model
   (`weights/kn5_numbers.binary`), same `lmplz`/`build_binary` pipeline as the main KN5.
4. At inference: score candidates `"1"`, `"11"`, ..., `"1"*8` under this model, argmax —
   same shape as the general n-gram's digit-length prediction, just a model trained
   exclusively on number contexts instead of everything.

| number scorer | accuracy (2023-row set, matches `num/`'s classifier comparison) |
|---|---|
| general n-gram (`ngram_4_a.bin`) | 71.97% (1456/2023) |
| **Gigaword+train number KN5** | **74.59% (1509/2023)** |

**+2.62pt** on number, on the full (non-sampled) number subset. On the 9448-row pooled
eval sample (202 number rows): 68.32% (138/202) → **73.27% (148/202)**, +4.95pt — overall
77.18% → **77.29%** (word/symbol untouched, this only replaces number routing).

Also worth noting: `num/`'s own earlier attempt at a number-specific model (bag-of-words
logreg/Naive Bayes, trained on the *same* `train_final.src.tok` the general n-gram
already sees) scored 48-54% on this same 2023-row set — *worse* than the general
n-gram. Confirms the lesson: a model "specific to number" only helps if it draws on
genuinely new data (Gigaword here), not just a same-corpus subset with less to learn
from.

**Continued Qwen training + wider beam (k5→k10)** (status.md "To try", both landed
together): Qwen2.5-3B LoRA continued for 6h more on Kaggle (fresh chunk,
`SKIP_LINES=900_000`, warmup_steps=30 fix carried over) — solo greedy-generation
accuracy (same stratified 10% sample, same eval as the original fine-tune) jumped
69.72%→**73.15% word** (+3.43pt, full 8198-row sample, not noise). Loss curve: steady
decline 2.41→2.26 over 1140 logged steps, still trending down at the 6h cutoff.
Paired with widening the beam from k5 to k10 (`scripts/infer_topk.py --k 10`) for
candidate generation — error EDA had found 9.5% of word rows hit a hard candidate-set
ceiling at k5, so this targets that directly. Combined effect on the candidate
ceiling: word top5 84.02%→**87.57%**, top10 88.13% (same sample).

Re-ran the full 3-way blend (Qwen tf + Mistral tf + KN5, `scripts/score_candidates.py`
`--lm-source topk` — added to read `infer_topk.py`'s plain-word `top_predictions`
column, distinct from the `word:score` format the older `zeroshot`/`lora`/`tf` sources
use) on the widened candidate set:

| | word | overall |
|---|---|---|
| k5, old checkpoint | 74.64% (6119/8198) | 77.29% (7302/9448) |
| **k10, 6h-retrained checkpoint** | **75.09% (6156/8198)** | **77.68% (7339/9448)** |

**+0.45pt word, +0.39pt overall.** Two changes bundled at once (retrained checkpoint +
wider beam), not isolated — same caveat as the earlier beam-vs-teacher-forced
comparison. Notably **λ_qwen now outweighs λ_mistral (0.50 vs 0.40)**, flipped from
the previous 0.20-0.30 vs 0.50-0.60 split — the retrained checkpoint's solo accuracy
gain is large enough that the blend leans back on Qwen, not just on Mistral's
diversity. **New best pipeline result: 75.09% word / 77.68% overall.**

## Final pipeline breakdown — how the 77.68% is achieved

**Routing by category** (`categorize()` on the answer token — has a real letter →
word, all-digits → number, else → symbol):

| category | method | accuracy |
|---|---|---|
| symbol | deterministic — predict the given first letter itself | 98.76% (1035/1048) |
| number | number-only KN5 (`kn5_numbers.binary`, Gigaword+train, see above) | 73.27% (148/202) |
| word | 3-way blend (below), k10 beam, 6h-retrained checkpoint | 75.09% (6156/8198) |
| **overall** | | **77.68% (7339/9448)** |

Symbol and number never touch the LLMs — symbols are a ~99% ceiling with nothing to
gain (confirmed at EDA stage); numbers are the corpus's anonymized `1111`-style
placeholder digits, which no LLM ever generates (structural, not a capability gap),
so the n-gram — which conditions digit-length on context — handles that category
alone.

**Word category = 3-way blend.** For each row, three signals are combined into one
score per candidate word, and the highest-scoring candidate wins:

```
score(w) = λ_qwen · qwen_tf(w)  +  λ_mistral · mistral_tf(w)  +  (1 − λ_qwen − λ_mistral) · kn5(w)
```

- **`qwen_tf(w)`** — fine-tuned Qwen2.5-3B LoRA's teacher-forced full-word log-prob.
  One batched forward pass per row (no beam search) sums per-token log-prob for each
  candidate under teacher forcing — `scripts/score_candidates.py`.
- **`mistral_tf(w)`** — Mistral-7B-v0.1 base, zero-shot (no fine-tuning), teacher-forced
  the same way, over the *same* candidate set Qwen already fixed (no separate beam
  search or candidate generation for Mistral at all).
- **`kn5(w)`** — a modified-Kneser-Ney 5-gram (KenLM, `tools/lmplz`, trained on the
  full 130M-token `train_final.src.tok`) log-prob of the candidate given context.
- **Candidate set** — union of Qwen's own beam-search top5 and the raw-count n-gram's
  top10 words starting with the required letter (`infer_ngram.topk_by_letter`). This
  lets the blend recover a correct word neither LLM's beam ever generated, not just
  rerank a fixed list.
- **λ_qwen, λ_mistral** — grid-searched over the 2D simplex (0.1 steps, `λ_qwen +
  λ_mistral ≤ 1`), tuned per fold; landed at λ_qwen≈0.20-0.30, λ_mistral≈0.50-0.60
  (KN5 gets the ~0.1-0.3 remainder) — Mistral outweighs Qwen despite lower solo
  accuracy, see above.

**Eval protocol.** Stratified 10% sample of `dev_set_final.csv` (seed 42, same sample
used throughout this comparison track) — 9448 rows total: 8198 word / 1048 symbol /
202 number. Word-category accuracy is measured by **pooled stratified 5-fold
cross-validation**: the 8198 word rows are split into 5 folds; for each fold, λ_qwen
and λ_mistral are grid-searched on the other 4 folds, then applied to predict the
held-out fold. Every row gets exactly one out-of-fold prediction, so the reported
74.64% is a genuine held-out estimate over the full sample — not a single noisy
80/20 split (an earlier version of this pipeline reported 72.50% word from exactly
that mistake; the true pooled-CV number was 71.52%, see the n-gram-blend section
above). Symbol and number accuracy are computed directly (deterministic / n-gram-only,
no tuning needed, so no CV split required for those categories).

**Scripts**: `scripts/gen_matched_scores.py` (Qwen beam-k5 pass), `scripts/score_candidates.py`
(Qwen and Mistral teacher-forced scoring, `--lm-source lora|tf`), `scripts/blend3.py`
(3-way blend + grid search + pooled CV report, number routing via `--number-model`).
Data: `weights/qwen_tf_scores.csv`, `weights/mistral_tf_scores.csv`, `weights/kn5.binary`,
`weights/kn5_numbers.binary`, `weights/ngram_4_a.bin`.
