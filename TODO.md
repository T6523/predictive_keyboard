# TODO

- **Fix `predict_accuracy.py` ranking.** Currently argmaxes `BaseScore` (smoothed KN
  probability) over the full vocab filtered by first letter -- wrong objective for
  exact-match accuracy (optimizes perplexity, not "most likely literal continuation").
  Causes: (1) Kneser-Ney discounting can rank a common word above the true highest-count
  continuation, (2) unseen contexts back off toward global unigram frequency, ignoring
  whether the candidate ever followed this context.
  Fix: use `model.BaseFullScore(state, word, out).ngram_length` to get the actual matched
  order per candidate. Sort by `ngram_length` desc first (deepest real n-gram hit wins),
  `log_prob` only as tiebreak within the same order -- replicates `eval_ngram.py`'s
  per-order argmax-on-observed-counts logic on top of the KenLM trie instead of the raw
  count pickle. Re-run accuracy on devv_eval/devv_test for model_a and model_b after.
