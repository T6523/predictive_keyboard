# Report: KenLM Pipeline — Architecture, Experiments, Results

Run reference: `weights/run_20260905_103654/` (production models, still current).

## 1. Architecture (Part 1)

`pipeline.py` — deterministic training + inference pipeline.

- **Model A**: 5-gram KenLM, trained on `data/train.src.tok` (raw, unclean — punctuation/number
  tokens kept, lowercased, ~703MB).
- **Model B**: 5-gram KenLM, trained on `data/gigaword.tok` (raw, vocab-masked to Model A's
  vocab via `[UNK]`, ~7GB).
- Training: `lmplz -o 5 -S 12G --prune 0 0 1 --discount_fallback` → ARPA → `build_binary trie`
  → `.klm` (memory-mapped trie), ARPA deleted after compile.
- Inference: streams `data/devv_eval.csv` / `data/devv_test.csv` row-by-row (`csv.DictReader`),
  scores `context` column with both models' `.score()`, writes `model_a_logprob10` /
  `model_b_logprob10` columns. RAM stays flat (mmap + streaming, no full-file load).
- Reproducible: `lmplz` has no thread/seed nondeterminism in this build; same corpus bytes +
  same flags → byte-identical ARPA → byte-identical trie → identical scores, every run.
- Output: `weights/run_<timestamp>/{model_a.klm, model_b.klm, devv_eval_predictions.csv,
  devv_test_predictions.csv}`.

Production run: `weights/run_20260905_103654/` — `model_a.klm` 483MB, `model_b.klm` 4.37GB.

## 2. Next-word accuracy eval (`predict_accuracy.py`)

Separate task from Part 1's log-prob scoring: given `context` + `first letter`, predict the
exact next word, compare to `answer` column. Two routes:

- **Symbol route**: first letter non-alnum → deterministic rule (`scripts/symbol_predict.py`):
  `[` → `[UNK]`, anything else → itself. Pre-existing repo utility, verified ~100% on this data.
- **Alnum route**: first letter alnum → KenLM n-gram lookup. Candidate pool = `weights/vocab.txt`
  (98,993 alnum words, built from alnum-only corpus by `scripts/build_vocab.py`), filtered by
  first letter, ranked via `model.BaseFullScore(state, word, out)`: sort by `(ngram_length desc,
  log_prob desc)` — deepest actually-observed n-gram order wins, smoothed log-prob only breaks
  ties within the same order. Chosen over plain `BaseScore` argmax because Kneser-Ney smoothing
  can rank a common word above the true highest-count continuation for exact-match purposes.

## 3. Experiments and results

### 3a. Ranking logic: `BaseScore` argmax vs `ngram_length`-first (model_a, model_b, both splits)

| | BaseScore argmax (v1) | ngram_length-first (v2) |
|---|---|---|
| model_a eval | 51.61% | 51.42% |
| model_a test | 51.52% | 51.30% |
| model_b eval | 46.59% | 46.49% |
| model_b test | 46.78% | 46.67% |

**Result: no meaningful difference (~0.1-0.3pp).** KenLM's internal backoff already orders
candidates similarly to an order-first scheme in practice. Not the cause of the accuracy gap
vs. the prior pure-Python baseline (~55%).

### 3b. Corpus/pruning hypotheses (model_a only, eval split, isolated one-variable-at-a-time)

Baseline for comparison: model_a eval, old code (no symbol fix), 51.42%.

| experiment | change | corpus | accuracy | delta |
|---|---|---|---|---|
| no pruning | `--prune 0 0 0` instead of `0 0 1` | raw `train.src.tok` | 51.71% | +0.29pp |
| alnum-cleaned | same prune, corpus stripped to alnum-only | `clean/train.src.tok` | 45.19% | **-6.23pp** |

**Pruning hypothesis: dead.** Keeping singleton 3/4/5-grams barely moves accuracy — KenLM's
backoff already recovers most of what pruning would have removed via lower-order fallback.

**Cleaned-corpus hypothesis: dead, and actively harmful.** `devv_eval.csv` context strings
still contain punctuation tokens (~9% token density, matching raw corpus's ~11% — confirmed by
direct token count). A model trained on punctuation-stripped text treats every punctuation
token in the eval context as OOV, corrupting the KenLM state used to prime prediction. Training
vocabulary must match eval context format; the raw/unclean corpus is the *correct* choice for
Part 1, not a bug.

### 3c. Symbol-route wiring (the actual fix)

`predict_accuracy.py` originally never called `symbol_predict.py` — symbol-letter rows (~10.7%
of every split, 8148/75860 eval, 2037/18965 test) fell through the alnum candidate lookup,
found nothing, and were silently scored wrong. Wiring in `is_symbol_letter`/`predict_symbol`
before the alnum branch fixed this.

| model | split | overall (before) | overall (after) | alnum-only | symbol-only |
|---|---|---|---|---|---|
| model_a | eval | 51.42% | 62.10% | **57.61%** | 99.37% |
| model_a | test | 51.30% | 61.98% | **57.47%** | 99.41% |
| model_b | eval | 46.49% | 57.16% | **52.08%** | 99.37% |
| model_b | test | 46.67% | 57.35% | **52.29%** | 99.41% |

**This was the real gap.** ~10.7pp of overall accuracy was purely a missing deterministic rule,
unrelated to the KenLM model or ranking logic. Model A's true (alnum-only) accuracy, 57.6%,
now exceeds the prior pure-Python baseline (~55%).

## 3d. Part 2: interpolation of Model A + Model B (log-prob / perplexity)

`interpolate.py`. Standard linear interpolation in probability space: `P = λ·10^logA +
(1-λ)·10^logB`, perplexity `= 10^(-Σlog10(P)/word_count)`. Reuses `model_a_logprob10` /
`model_b_logprob10` columns already written by `pipeline.py` — no rescoring, pure CSV
arithmetic. λ tuned by grid search (step 0.01) minimizing perplexity on `devv_eval`, then
applied as-is to `devv_test`.

| | eval perplexity | test perplexity |
|---|---|---|
| model A alone (λ=1) | 92.337 | 92.214 |
| model B alone (λ=0) | 150.597 | 150.503 |
| **interpolated (λ=0.71)** | **73.956** | **74.000** |

Best λ (0.71, tuned on eval) generalizes cleanly to test (73.956 → 74.000, no overfitting).
Interpolation beats both solo models on both splits — ~20% perplexity reduction vs. Model A
alone. Confirms Model A (in-domain) and Model B (gigaword) carry complementary information
worth combining, even though A dominates the mixture weight.

### 3e. Interpolation applied to next-word accuracy

`predict_accuracy.py --model-b ... --lam 0.71` (new `predict_interp`): per candidate, prime a
separate context state per model, argmax over `λ·10^logA + (1-λ)·10^logB` (probability-space
interpolation, same formula as `interpolate.py`, applied per-candidate instead of per-sentence).
Reused the perplexity-tuned λ=0.71 as-is, no separate accuracy-specific tuning yet. Symbol
route unaffected (deterministic rule, no model involved).

| | eval alnum-only | test alnum-only |
|---|---|---|
| model A solo | 57.61% | 57.47% |
| model B solo | 52.08% | 52.29% |
| **interpolated (λ=0.71)** | **60.16%** | **60.02%** |

+2.5-2.6pp over Model A alone, consistent across both splits, no overfitting (eval and test
track each other closely, same as the perplexity result). Confirms the perplexity-domain
interpolation gain transfers to the accuracy objective even without re-tuning λ specifically
for it.

**Not yet done:** grid-searching a separate accuracy-tuned λ (current 0.71 was tuned for
perplexity, not accuracy — a different optimum may do slightly better, at the cost of one
`predict_accuracy.py` pass per grid point, ~2800s eval / ~220s test each).

## 4. Current state

- Production models (`weights/run_20260905_103654/model_a.klm`, `model_b.klm`) trained on raw
  (unclean) corpora — confirmed correct per §3b.
- `predict_accuracy.py` fixed: `ngram_length`-first ranking + symbol-route wiring +
  alnum/symbol accuracy breakdown reporting.
- Model A (in-domain) beats Model B (gigaword, masked vocab) by ~5.4pp alnum accuracy on both
  splits — expected, reflects training-domain distance.
- Scratch experiment dirs (`weights/experiments/exp_a_noprune`, `exp_b_clean`, ~2GB) deleted
  after comparison — findings preserved above, binaries not needed going forward.
- Part 2 (interpolating Model A + Model B) not yet started.

## 5. Ideas not yet tried (future work)

1. **Part 2 interpolation** — weighted log-prob combination or per-candidate best-of-both
   ranking (prefer whichever model has the deeper observed `ngram_length`). Likely biggest
   remaining lever; A and B disagree on a meaningful fraction of rows.
2. **Higher order** (6-7 gram instead of 5) — cheap to test on model_a alone, same isolation
   method as §3b.
3. **True stupid-backoff scoring** (`0.4^k` discount per order, matching `eval_ngram.py`'s
   convention) instead of `ngram_length`-first tie-break — closer to the pure-count baseline's
   exact formula.
