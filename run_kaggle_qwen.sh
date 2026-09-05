#!/usr/bin/env bash
# Push train_and_eval.py (Qwen2.5-0.5B LoRA fine-tune + constrained eval) to Kaggle as a script
# kernel, poll till it finishes, pull the adapter checkpoint + prediction csvs back.
#
# Differs from transformer/run_kaggle.sh in the two ways that actually matter:
#   - kernel_type "script" (single .py file), not "notebook" -- train_and_eval.py doesn't need
#     the write-my-dependencies-as-strings trick build_notebook.py uses, it has none left
#     (scripts/symbol_predict.py's two functions are inlined in it for exactly this reason).
#   - enable_internet: true -- needs it for `pip install unsloth trl peft bitsandbytes` and the
#     HF Hub download of the base model. The GPT2/Qwen2-from-scratch kernel runs with internet
#     off; this one can't.
#
# No resume-across-sessions support (unlike transformer/run_kaggle.sh) -- train_and_eval.py
# trains once for one time-budget then evals, it doesn't yet load a prior adapter to continue
# from. Add that to train_and_eval.py first if you need it; this script only does push/poll/pull.
#
# Usage: run_kaggle_qwen.sh [--data-dataset SLUG] [--vocab-dataset SLUG]
#   --data-dataset SLUG    train.src.tok, devv_eval.csv, devv_test.csv (default: teekn07/keyboard)
#   --vocab-dataset SLUG   vocab.txt only -- kept separate so the ~700MB data dataset never needs
#                          reuploading just to add one 800KB file (Kaggle's `datasets version`
#                          replaces a dataset's whole file set, it's not additive; a second small
#                          dataset avoids that). default: teekn07/predictive-keyboard-vocab
set -euo pipefail
cd "$(dirname "$0")"

DATA_DATASET="teekn07/keyboard"
VOCAB_DATASET="teekn07/predictive-keyboard-vocab"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dataset) DATA_DATASET="$2"; shift 2 ;;
    --vocab-dataset) VOCAB_DATASET="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

export PATH="$HOME/.local/bin:$PATH"
USERNAME=$(kaggle config view 2>&1 | sed -n 's/.*username: //p')

cp train_and_eval.py qwen_kernel/train_and_eval.py
sed -e "s#{USERNAME}#$USERNAME#" -e "s#{DATASET_SOURCES}#\"$DATA_DATASET\", \"$VOCAB_DATASET\"#" \
  qwen_kernel/kernel-metadata.template.json > qwen_kernel/kernel-metadata.json
echo "--- kernel-metadata.json ---"; cat qwen_kernel/kernel-metadata.json

echo "--- pushing ---"
kaggle kernels push -p qwen_kernel

SLUG="$USERNAME/predictive-keyboard-qwen"
echo "--- polling $SLUG (60s interval, ~10h cap -- 8h training budget + full-set eval on top) ---"
for i in $(seq 1 600); do
  STATUS=$(kaggle kernels status "$SLUG" 2>&1)
  echo "[$i] $STATUS"
  case "$STATUS" in
    *COMPLETE*) break ;;
    *ERROR*|*CANCEL*) echo "kernel run failed"; exit 1 ;;
  esac
  sleep 60
done

echo "--- pulling output ---"
rm -rf qwen_output
kaggle kernels output "$SLUG" -p qwen_output
ls -la qwen_output
