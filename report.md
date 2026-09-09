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

Weights: **[mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)** —
open weight, Apache 2.0, no gated access.

**Why:** need >80% accuracy ceiling on a benchmark close to this task's shape
(context-conditioned next-word prediction), fitting an 8GB VRAM RTX 4060 Laptop for
training. HellaSwag (0-shot) 84.0% self-reported, 83.2% independently reproduced via
EleutherAI's eval harness — two-source agreement, no outlier vs. its ARC-Challenge
(60.0%) / MMLU (60.1%) scores, so the number isn't a contamination fluke. 7B params
fits via 4-bit QLoRA (~4-4.5GB weights + adapter/optimizer overhead) on this machine,
same fine-tune path already used for `qwen/` LoRA. Runner-up Qwen2.5-7B and
Llama-3.1-8B also clear 80% and are open weight, but had noisier/less cross-validated
scraped numbers at pick time — re-check before swapping in.

Still open: HellaSwag isn't this task (no first-letter constraint, multiple-choice not
open-vocab) — treat the 80%+ as a backbone-quality filter, not a predicted score on
`dev_set_final.csv`. Actual accuracy must be measured post-fine-tune the same way as
the n-gram baseline (`scripts/infer_ngram.py`-style harness).
