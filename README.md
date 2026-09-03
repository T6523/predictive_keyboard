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
clean/train.src.tok  ──►  train_ngram.py  ──►  weights/ngram_N.bin
```

1. `scripts/split_dev.py` — split `dev_set.csv` 80/20 into `devv_eval.csv` (tuning) / `devv_test.csv` (holdout).
2. `scripts/clean_pipeline.py` — lowercase, fix `- lrb -` → `-lrb-` style bracket tokens.
3. `scripts/clean_alnum.py` — strip non-alnum tokens from the training corpus and from csv `context` columns.
4. `scripts/train_ngram.py --n {3,4,5} --out weights/ngram_N.bin` — count-based n-gram model (orders 1..n, no smoothing baked in).
5. `scripts/eval_ngram.py --model weights/ngram_N.bin --data clean/devv_eval.csv` — score the n-gram alone (stupid backoff: highest order first, keep candidates starting with the required letter, fall back on miss).
6. `scripts/eval.py --data clean/devv_test.csv --model weights/ngram_N.bin [--out preds.csv]` — full pipeline: symbol/[UNK] rows answered deterministically, rest through the n-gram; reports overall + per-route accuracy.

## Layout

```
data/     raw csv/tok files (untouched originals)
clean/    cleaned outputs of clean_pipeline.py + clean_alnum.py
weights/  trained ngram_N.bin model files
eda/      exploration notebooks
scripts/  pipeline scripts (see above)
```

## Usage

```bash
python3 scripts/split_dev.py
python3 scripts/clean_pipeline.py
python3 scripts/clean_alnum.py
python3 scripts/train_ngram.py --n 3 --train clean/train.src.tok --out weights/ngram_3.bin
python3 scripts/eval_ngram.py --model weights/ngram_3.bin --data clean/devv_eval.csv
python3 scripts/eval.py --data clean/devv_test.csv --model weights/ngram_3.bin
```

## Notes

- Model files (`weights/*.bin`) and raw/cleaned corpora are large (100MB-600MB+) — kept out of git, see `.gitignore`. Regenerate with `train_ngram.py`.
- Training a higher-order n-gram (n=4, n=5) on the full corpus is memory-heavy — context space grows combinatorially with order. Watch RAM during `train_ngram.py --n 4/5`.
