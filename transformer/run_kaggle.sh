#!/usr/bin/env bash
# Push transformer/kernel/train.ipynb to Kaggle, poll till it finishes, pull the checkpoint
# back, and persist it as a Kaggle dataset so the *next* run can resume from it.
#
# Usage: transformer/run_kaggle.sh [--ckpt-dataset SLUG] [--data-dataset SLUG]
#   --ckpt-dataset SLUG   Kaggle dataset (your account) to push checkpoints/*.pt into after
#                         each run. Created on first use. Pass the same slug next time to
#                         auto-attach it as a resume input.
#   --data-dataset SLUG   training-data dataset to attach (default: krittiteen/gigaword)
set -euo pipefail
cd "$(dirname "$0")/kernel"

CKPT_DATASET=""
DATA_DATASET="krittiteen/gigaword"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt-dataset) CKPT_DATASET="$2"; shift 2 ;;
    --data-dataset) DATA_DATASET="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

export PATH="$HOME/.local/bin:$PATH"
USERNAME=$(kaggle config view 2>&1 | sed -n 's/.*username: //p')
SOURCES="\"$DATA_DATASET\""
if [[ -n "$CKPT_DATASET" ]] && kaggle datasets status "$CKPT_DATASET" >/dev/null 2>&1; then
  SOURCES="$SOURCES, \"$CKPT_DATASET\""
  echo "resuming: attaching checkpoint dataset $CKPT_DATASET"
fi

sed -e "s#{USERNAME}#$USERNAME#" -e "s#{DATASET_SOURCES}#$SOURCES#" \
  kernel-metadata.template.json > kernel-metadata.json
echo "--- kernel-metadata.json ---"; cat kernel-metadata.json

echo "--- pushing ---"
# no --accelerator flag: it overrides kernel-metadata.json's machine_shape with its own literal
# value, and "gpu" is not one of the 3 valid strings (NvidiaTeslaT4/NvidiaTeslaP100/Tpu1VmV38) --
# passing it clobbers a correctly-set machine_shape and silently falls back to random allocation.
kaggle kernels push -p .

SLUG="$USERNAME/predictive-keyboard-transformer"
echo "--- polling $SLUG (20s interval, ~40min cap) ---"
for i in $(seq 1 120); do
  STATUS=$(kaggle kernels status "$SLUG" 2>&1)
  echo "[$i] $STATUS"
  case "$STATUS" in
    *COMPLETE*) break ;;
    *ERROR*|*CANCEL*) echo "kernel run failed"; exit 1 ;;
  esac
  sleep 20
done

echo "--- pulling output ---"
rm -rf output
kaggle kernels output "$SLUG" -p output
ls -la output

if [[ -n "$CKPT_DATASET" ]] && compgen -G "output/checkpoints/*.pt" >/dev/null; then
  echo "--- persisting checkpoint -> $CKPT_DATASET ---"
  if kaggle datasets status "$CKPT_DATASET" >/dev/null 2>&1; then
    kaggle datasets version -p output -m "run $(date -Iseconds)" -r zip
  else
    echo "dataset doesn't exist yet, creating it"
    python3 - "$CKPT_DATASET" output <<'PY'
import json, sys
slug, path = sys.argv[1], sys.argv[2]
json.dump({"title": slug.split("/", 1)[1], "id": slug, "licenses": [{"name": "CC0-1.0"}]},
          open(f"{path}/dataset-metadata.json", "w"))
PY
    kaggle datasets create -p output -r zip
  fi
  echo "next run: transformer/run_kaggle.sh --ckpt-dataset $CKPT_DATASET  (auto-resumes)"
fi
