#!/usr/bin/env bash
# Push infer_kaggle.py (GPT2 checkpoint inference, no training) to Kaggle as a script kernel,
# poll till done, pull the prediction csvs back. No new upload needed -- both datasets already
# have what this needs (checkpoint in teekn07/predictive-keyboard-ckpt, devv csvs in teekn07/keyboard).
#
# Local inference (33k_full_gpt/infer.py) stalled at <1.5 rows/s, ~10x under its own
# smoke-test rate, cause unconfirmed (CPU-bound at 98% single-core despite GPU also pegged --
# looks like per-row kernel-launch/Python overhead, not compute). Kaggle T4 isn't necessarily
# faster raw, but the run time is at least predictable.
#
# Usage: gpt/run_kaggle_infer.sh [--ckpt-dataset SLUG] [--data-dataset SLUG]
set -euo pipefail
cd "$(dirname "$0")"

CKPT_DATASET="teekn07/predictive-keyboard-ckpt"
DATA_DATASET="teekn07/keyboard"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt-dataset) CKPT_DATASET="$2"; shift 2 ;;
    --data-dataset) DATA_DATASET="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

export PATH="$HOME/.local/bin:$PATH"
USERNAME=$(kaggle config view 2>&1 | sed -n 's/.*username: //p')

sed -e "s#{USERNAME}#$USERNAME#" -e "s#{DATASET_SOURCES}#\"$CKPT_DATASET\", \"$DATA_DATASET\"#" \
  infer_kernel/kernel-metadata.template.json > infer_kernel/kernel-metadata.json
echo "--- kernel-metadata.json ---"; cat infer_kernel/kernel-metadata.json

echo "--- pushing ---"
kaggle kernels push -p infer_kernel

SLUG="$USERNAME/predictive-keyboard-gpt-infer"
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
rm -rf infer_output
kaggle kernels output "$SLUG" -p infer_output
ls -la infer_output
