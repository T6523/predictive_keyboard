#!/usr/bin/env bash
# Unattended tail of the pipeline: waits out the running Stage 1 Kaggle kernel,
# merges it with the already-done local Stage 1 slice, ships that as Stage 2's
# input dataset, runs Stage 2 (Mistral, dual-GPU), then Stage 3 (blend) --
# no further prompting needed. Safe to nohup+disown; only needs WSL to stay up
# (screen lock is fine, actual sleep/hibernate is not -- see status.md).
#
# Usage: nohup bash run_remaining_pipeline.sh > /tmp/pipeline_remaining.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

QWEN_KERNEL=krittiteen/qwen-only-infer
MISTRAL_KERNEL=krittiteen/mistral-only-infer
DATASET_SLUG=krittiteen/stage1-qwen-scores-full
MISTRAL_PUSH_DIR="/tmp/claude-1000/-home-asus-projects-nlp-comp1-predictive-keyboard/70f4f2a4-da4b-4e8a-b3f4-3418cb5dfc7d/scratchpad/kaggle_push_mistral_only"
KAGGLE_DL_DIR=../weights/stage1_kaggle
MERGED_DIR=../weights/stage1_merged
DATASET_STAGE_DIR=../weights/stage1_dataset_stage
MISTRAL_DL_DIR=../weights/stage2

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# --- wait for a kernel to leave RUNNING/QUEUED, tolerating transient API errors ---
wait_for_kernel() {
  local slug=$1
  local prev=""
  while true; do
    local s
    s=$(kaggle kernels status "$slug" 2>&1)
    if echo "$s" | grep -qiE "error|unavailable|timeout"; then
      sleep 20; continue
    fi
    [ "$s" != "$prev" ] && log "$slug: $s" && prev="$s"
    echo "$s" | grep -qi "running\|queued" || { echo "$s"; return 0; }
    sleep 60
  done
}

fail() { log "FATAL: $*"; exit 1; }

log "=== waiting for Stage 1 Kaggle kernel ($QWEN_KERNEL) ==="
final=$(wait_for_kernel "$QWEN_KERNEL")
echo "$final" | grep -qi "complete" || fail "Stage 1 kernel ended as: $final"

log "=== downloading Stage 1 Kaggle output ==="
rm -rf "$KAGGLE_DL_DIR"; mkdir -p "$KAGGLE_DL_DIR"
kaggle kernels output "$QWEN_KERNEL" -p "$KAGGLE_DL_DIR" 2>&1 | tail -20
grep -qE "Traceback \(most recent call last\)|CUDA out of memory|SIGKILL" "$KAGGLE_DL_DIR/qwen-only-infer.log" && fail "Stage 1 kernel log has errors -- check $KAGGLE_DL_DIR/qwen-only-infer.log"
[ -f "$KAGGLE_DL_DIR/dev_qwen_scores_kaggle.jsonl" ] || fail "missing dev_qwen_scores_kaggle.jsonl in Stage 1 output"
[ -f "$KAGGLE_DL_DIR/test_qwen_scores_kaggle.jsonl" ] || fail "missing test_qwen_scores_kaggle.jsonl in Stage 1 output"

log "=== merging local + Kaggle Stage 1 slices ==="
python3 merge_stage1_shards.py \
  --local-dev ../weights/stage1/dev_qwen_scores.jsonl \
  --local-test ../weights/stage1/test_qwen_scores.jsonl \
  --kaggle-dev "$KAGGLE_DL_DIR/dev_qwen_scores_kaggle.jsonl" \
  --kaggle-test "$KAGGLE_DL_DIR/test_qwen_scores_kaggle.jsonl" \
  --out-dir "$MERGED_DIR" || fail "merge_stage1_shards.py failed"

log "=== updating Stage 1 dataset for Stage 2 to consume ==="
cp "$MERGED_DIR/dev_qwen_scores.jsonl" "$DATASET_STAGE_DIR/"
cp "$MERGED_DIR/test_qwen_scores.jsonl" "$DATASET_STAGE_DIR/"
(cd "$DATASET_STAGE_DIR" && kaggle datasets version -p . -m "full merged stage1 output" -r zip) 2>&1 || fail "dataset version update failed"
sleep 30  # let Kaggle index the new version before referencing it

log "=== pushing Stage 2 Mistral kernel ($MISTRAL_KERNEL) ==="
(cd "$MISTRAL_PUSH_DIR" && kaggle kernels push -p .) 2>&1 || fail "mistral kernel push failed"

log "=== waiting for Stage 2 Kaggle kernel ==="
final2=$(wait_for_kernel "$MISTRAL_KERNEL")
echo "$final2" | grep -qi "complete" || fail "Stage 2 kernel ended as: $final2"

log "=== downloading Stage 2 Mistral output ==="
rm -rf "$MISTRAL_DL_DIR"; mkdir -p "$MISTRAL_DL_DIR"
kaggle kernels output "$MISTRAL_KERNEL" -p "$MISTRAL_DL_DIR" 2>&1 | tail -20
grep -qE "Traceback \(most recent call last\)|CUDA out of memory|SIGKILL" "$MISTRAL_DL_DIR/mistral-only-infer.log" && fail "Stage 2 kernel log has errors -- check $MISTRAL_DL_DIR/mistral-only-infer.log"
[ -f "$MISTRAL_DL_DIR/dev_mistral_scores.jsonl" ] || fail "missing dev_mistral_scores.jsonl in Stage 2 output"
[ -f "$MISTRAL_DL_DIR/test_mistral_scores.jsonl" ] || fail "missing test_mistral_scores.jsonl in Stage 2 output"

log "=== Stage 3: blending ==="
python3 combine_final.py \
  --dev-qwen "$MERGED_DIR/dev_qwen_scores.jsonl" \
  --dev-mistral "$MISTRAL_DL_DIR/dev_mistral_scores.jsonl" \
  --test-qwen "$MERGED_DIR/test_qwen_scores.jsonl" \
  --test-mistral "$MISTRAL_DL_DIR/test_mistral_scores.jsonl" \
  --out-test-pred ../weights/test_set_pred.txt 2>&1 | tee ../weights/stage3_final_report.txt
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "combine_final.py failed -- see ../weights/stage3_final_report.txt"

log "=== DONE -- predictions at ../weights/test_set_pred.txt ==="
