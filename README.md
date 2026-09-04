# Predictive Keyboard — N-gram Language Model

Contest 1 (Language Modeling): predict the next word given a context and its first letter.

## Approach

- **Symbol / `[UNK]` first letters** → answered deterministically (`scripts/symbol_predict.py`), no model needed. `[` always means `[UNK]`; any other non-alnum first letter maps to itself.
- **Alnum first letters** → routed to an n-gram language model with stupid backoff.

## Pipeline

```
data/train.src.tok
      │  clean_pipeline.py   (fix PTB bracket tokens, lowercase)
      ▼
      │  clean_alnum.py      (drop non-alnum tokens; keep all csv rows)
      ▼
clean/train.src.tok  ──►  lmplz  ──►  weights/ngram_N.arpa  ──►  build_binary  ──►  weights/ngram_N.bin
```

1. `scripts/split_dev.py` — split `dev_set.csv` 80/20 into `devv_eval.csv` (tuning) / `devv_test.csv` (holdout).
2. `scripts/clean_pipeline.py` — lowercase, fix `- lrb -` → `-lrb-` style bracket tokens.
3. `scripts/clean_alnum.py` — strip non-alnum tokens from the training corpus and from csv `context` columns.
4. `scripts/bin/lmplz -o N < clean/train.src.tok > weights/ngram_N.arpa` — trains the count-based n-gram (orders 1..N, modified Kneser-Ney smoothing). [KenLM](https://kheafield.com/code/kenlm/)'s C++ trainer — swapped in for the original pure-Python `train_ngram.py` to fix the RAM blowup on higher orders (Python's per-object overhead on millions of count-table entries vs. KenLM's packed structures). `lmplz` prints a live stage-by-stage progress bar (counting → sorting → discounting → interpolating) to stderr.
5. `scripts/bin/build_binary weights/ngram_N.arpa weights/ngram_N.bin` — compiles the text `.arpa` into KenLM's mmap'd binary format (loads near-instantly, doesn't pull the whole model into RAM).
6. `scripts/eval_ngram.py --model weights/ngram_N.bin --data clean/devv_eval.csv` — score the n-gram alone (stupid backoff: highest order first, keep candidates starting with the required letter, fall back on miss).
7. `scripts/eval.py --data clean/devv_test.csv --model weights/ngram_N.bin [--out preds.csv]` — full pipeline: symbol/[UNK] rows answered deterministically, rest through the n-gram; reports overall + per-route accuracy.

`scripts/train_ngram.py` (pure Python, pickle-based) is kept for reference/small experiments but is no longer the primary path — `weights/*.bin` files are now KenLM binaries, not pickles.

## Transformer (transformer/)

A word-level GPT2/Qwen2 model (`transformer/models.py`), trained on Kaggle (`transformer/run_kaggle.sh` → `transformer/kernel/build_notebook.py`). Vocab (`transformer/vocab.py`) is built from whatever `train.src.tok` is attached as the Kaggle input dataset — in practice the **raw** `data/train.src.tok` (99021 unique tokens, symbols kept), not `clean/train.src.tok`. `MIN_COUNT=1` (keep everything) + `<s>`/`</s>` added at index 0/1 → checkpoint `vocab_size` is **99023**.

## gigaword.tar.gz preprocessing

`data/gigaword.tar.gz` is the raw NYT Gigaword archive (149 monthly files, 1994-07..2006-12, 6.3GB uncompressed) the training corpus's domain comes from — not itself training data, useful for vocab-coverage and pretraining experiments.

- `eda/gigaword_eda.ipynb` — streams the tar.gz directly (never extracted to disk), one pass: paragraph length distribution, token category/case breakdown, Zipf plot, vocab overlap with `train.src.tok`/dev/test, and a cumulative-coverage table (vocab size needed for 95/97.5/99/99.99% token coverage). Caches the full-corpus pass to `data/gigaword_stats_cache.pkl` (~10min first run, instant reruns).
- `scripts/tokenize_gigaword.py` — tokenizes gigaword onto `train.src.tok`'s exact vocab: lowercases, isolates punctuation the same way `train.src.tok` does (`o.j.` → `o . j .`, `don't` → `don ' t`, `-LRB-` → `- lrb -`), masks any token outside the 99021-word vocab with `[UNK]`. Nothing stripped (unlike `clean_alnum.py` — symbols/`[UNK]` all survive as tokens). Streams from the tar.gz, writes to `data/gigaword.tok` (not `clean/`, since it's a derived-from-raw artifact, not a cleaned version of an existing `clean/` file). `--limit N` for a quick test run.

## Layout

```
data/        raw csv/tok files (untouched originals) + gigaword.tar.gz / gigaword.tok
clean/       cleaned outputs of clean_pipeline.py + clean_alnum.py
weights/     trained ngram_N.bin model files
transformer/ word-level GPT2/Qwen2 model + Kaggle training kernel
eda/         exploration notebooks
scripts/     pipeline scripts (see above)
```

## Usage

```bash
python3 scripts/split_dev.py
python3 scripts/clean_pipeline.py
python3 scripts/clean_alnum.py

# build lmplz + build_binary once (needs cmake, libboost-all-dev, libeigen3-dev,
# zlib1g-dev, libbz2-dev, liblzma-dev — see scripts/build_kenlm.sh)
scripts/bin/lmplz -o 3 < clean/train.src.tok > weights/ngram_3.arpa
scripts/bin/build_binary weights/ngram_3.arpa weights/ngram_3.bin
rm weights/ngram_3.arpa  # intermediate text file, regenerable, ~2x the .bin size

python3 scripts/eval_ngram.py --model weights/ngram_3.bin --data clean/devv_eval.csv
python3 scripts/eval.py --data clean/devv_test.csv --model weights/ngram_3.bin
```

## Notes

- Model files (`weights/*.bin`) and raw/cleaned corpora are large (100MB-600MB+) — kept out of git, see `.gitignore`. Regenerate with `train_ngram.py`.
- Training a higher-order n-gram (n=4, n=5) on the full corpus is memory-heavy — context space grows combinatorially with order. Watch RAM during `train_ngram.py --n 4/5`.
