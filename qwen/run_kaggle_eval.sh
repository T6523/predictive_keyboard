#!/usr/bin/env bash
# Push eval_qwen.py (LoRA adapter + constrained eval, no training) to Kaggle as a script kernel,
# poll till done, pull the prediction csvs back. Companion to run_kaggle.sh's full
# train+eval run -- use this once an adapter is already saved (train+eval's own eval pass OOM'd,
# see eval_qwen.py's docstring) to retry just the eval step, no retraining.
#
# Usage: qwen/run_kaggle_eval.sh [--adapter-dataset SLUG] [--data-dataset SLUG] [--vocab-dataset SLUG] [--kenlm-dataset SLUG]
#   --adapter-dataset SLUG   adapter_config.json + adapter_model.safetensors + tokenizer.* at its
#                            root. default: teekn07/predictive-keyboard-qwen-adapter (placeholder --
#                            upload the real adapter files there before running this)
#   --data-dataset SLUG      devv_eval.csv, devv_test.csv. default: teekn07/keyboard
#   --vocab-dataset SLUG     vocab.txt. default: teekn07/predictive-keyboard-vocab
#   --kenlm-dataset SLUG     model_a.klm (5-gram, used to shortlist candidates before Qwen scores
#                            them -- see eval_qwen.py's docstring). default: teekn07/predictive-keyboard-kenlm-a
set -euo pipefail
cd "$(dirname "$0")"

ADAPTER_DATASET="teekn07/predictive-keyboard-qwen-adapter"
DATA_DATASET="teekn07/keyboard"
VOCAB_DATASET="teekn07/predictive-keyboard-vocab"
KENLM_DATASET="teekn07/predictive-keyboard-kenlm-a"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapter-dataset) ADAPTER_DATASET="$2"; shift 2 ;;
    --data-dataset) DATA_DATASET="$2"; shift 2 ;;
    --vocab-dataset) VOCAB_DATASET="$2"; shift 2 ;;
    --kenlm-dataset) KENLM_DATASET="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

export PATH="$HOME/.local/bin:$PATH"
USERNAME=$(kaggle config view 2>&1 | sed -n 's/.*username: //p')

sed -e "s#{USERNAME}#$USERNAME#" \
    -e "s#{DATASET_SOURCES}#\"$ADAPTER_DATASET\", \"$DATA_DATASET\", \"$VOCAB_DATASET\", \"$KENLM_DATASET\"#" \
  eval_kernel/kernel-metadata.template.json > eval_kernel/kernel-metadata.json
echo "--- kernel-metadata.json ---"; cat eval_kernel/kernel-metadata.json

echo "--- pushing ---"
kaggle kernels push -p eval_kernel

SLUG="$USERNAME/predictive-keyboard-qwen-eval"
echo "--- polling $SLUG (30s interval) ---"
for i in $(seq 1 240); do
  STATUS=$(kaggle kernels status "$SLUG" 2>&1)
  echo "[$i] $STATUS"
  case "$STATUS" in
    *COMPLETE*) break ;;
    *ERROR*|*CANCEL*) echo "kernel run failed"; exit 1 ;;
  esac
  sleep 30
done

echo "--- pulling output ---"
rm -rf eval_output
kaggle kernels output "$SLUG" -p eval_output
ls -la eval_output
