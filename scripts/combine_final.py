"""Stage 3: combines Stage 1's (qwen_scores, kn5_scores) with Stage 2's
(mistral_scores) by row order, tunes (lam_qwen, lam_mistral) on a stratified first
half of dev, reports honestly on the second half, applies the frozen lambda to the
full test set and writes test_set_pred.txt (one word/line, no header, per the contest
PDF spec).

Stage 1 (scripts/run_qwen_local.py, local 4060) and Stage 2
(kaggle/mistral_only_infer.ipynb, Kaggle) write rows in the SAME order as their shared
input CSV and never reorder/filter/drop rows -- zipping by index is safe. A length
check (not a context check -- stage 2's output doesn't carry context) catches a
mismatched pair of files.

Usage:
    python3 combine_final.py --dev-qwen ../weights/stage1/dev_qwen_scores.jsonl \
        --dev-mistral ../weights/stage2/dev_mistral_scores.jsonl \
        --test-qwen ../weights/stage1/test_qwen_scores.jsonl \
        --test-mistral ../weights/stage2/test_mistral_scores.jsonl \
        --out-test-pred ../weights/test_set_pred.txt
"""
import argparse
import json
import random
import re

CAT_NUMBER = re.compile(r"[0-9]+")
CAT_WORD = re.compile(r"(?=.*[a-z])[a-z']+", re.IGNORECASE)


def categorize(tok):
    if CAT_NUMBER.fullmatch(tok):
        return "number"
    if CAT_WORD.fullmatch(tok):
        return "word"
    return "symbol"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def combine(qwen_rows, mistral_rows):
    """Zips Stage 1 + Stage 2 output by index into the same (r, triples, ngram_best)
    shape blend3.py's pick_best/eval_word expect -- triples None for non-word rows."""
    assert len(qwen_rows) == len(mistral_rows), (
        f"row count mismatch: qwen={len(qwen_rows)} mistral={len(mistral_rows)} -- "
        "these must be the SAME Stage 1 run's dev/test outputs")
    combined = []
    for qr, mr in zip(qwen_rows, mistral_rows):
        if qr["route"] != "word" or not qr.get("candidates"):
            combined.append((qr, None))
            continue
        cands = qr["candidates"]
        q_scores, kn5_scores = qr["qwen_scores"], qr["kn5_scores"]
        m_scores = mr["mistral_scores"]
        assert len(m_scores) == len(cands), "candidate count mismatch between stage1/stage2 for a row"
        triples = {w: (q_scores[i], m_scores[i], kn5_scores[i]) for i, w in enumerate(cands)}
        combined.append((qr, triples))
    return combined


def pick_best(triples, lam_q, lam_m):
    lam_n = 1 - lam_q - lam_m
    best_word, best_score = None, float("-inf")
    for w, (q, m, k) in triples.items():
        s = lam_q * q + lam_m * m + lam_n * k
        if s > best_score:
            best_word, best_score = w, s
    return best_word


def build_grid(steps):
    return [(i / (steps - 1), j / (steps - 1)) for i in range(steps) for j in range(steps - i)]


def eval_word(idx_list, combined, lam_q, lam_m):
    correct = total = 0
    for i in idx_list:
        r, triples = combined[i]
        if triples is None:
            continue
        pred = pick_best(triples, lam_q, lam_m)
        correct += (pred or "").strip().lower() == r["answer"].strip().lower()
        total += 1
    return correct, total


def stratified_half_split(combined, seed=42):
    rng = random.Random(seed)
    by_cat = {}
    for i, (r, _) in enumerate(combined):
        by_cat.setdefault(categorize(r["answer"]), []).append(i)
    halves = [[], []]
    for cat, idx in by_cat.items():
        idx = idx[:]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            halves[j % 2].append(i)
    return halves


def demo():
    """Self-check: pick_best argmax + combine()'s row-count guard, no files needed."""
    triples = {"a": (-2.0, -1.0, -1.0), "b": (-1.0, -2.0, -5.0)}
    assert pick_best(triples, 0.6, 0.4) == "b"
    assert pick_best(triples, 0.0, 0.0) == "a"

    qwen_rows = [{"route": "word", "candidates": ["a", "b"], "qwen_scores": [-2.0, -1.0],
                  "kn5_scores": [-1.0, -5.0], "answer": "b"},
                 {"route": "symbol", "pred": "!"}]
    mistral_rows = [{"mistral_scores": [-1.0, -2.0]}, {"mistral_scores": None}]
    combined = combine(qwen_rows, mistral_rows)
    assert combined[0][1] == {"a": (-2.0, -1.0, -1.0), "b": (-1.0, -2.0, -5.0)}
    assert combined[1][1] is None

    try:
        combine(qwen_rows, mistral_rows[:1])
        assert False, "should have raised on length mismatch"
    except AssertionError as e:
        assert "row count mismatch" in str(e)

    print("demo ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-qwen", default="../weights/stage1/dev_qwen_scores.jsonl")
    ap.add_argument("--dev-mistral", default="../weights/stage2/dev_mistral_scores.jsonl")
    ap.add_argument("--test-qwen", default="../weights/stage1/test_qwen_scores.jsonl")
    ap.add_argument("--test-mistral", default="../weights/stage2/test_mistral_scores.jsonl")
    ap.add_argument("--out-test-pred", default="../weights/test_set_pred.txt")
    ap.add_argument("--lam-steps", type=int, default=11)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    dev_combined = combine(load_jsonl(args.dev_qwen), load_jsonl(args.dev_mistral))
    test_combined = combine(load_jsonl(args.test_qwen), load_jsonl(args.test_mistral))
    print(f"{len(dev_combined)} dev rows, {len(test_combined)} test rows combined")

    tune_idx, report_idx = stratified_half_split(dev_combined)
    print(f"tune half: {len(tune_idx)} rows, report half: {len(report_idx)} rows")

    grid = build_grid(args.lam_steps)
    best_lams, best_acc = (1.0, 0.0), -1
    for lam_q, lam_m in grid:
        c, t = eval_word(tune_idx, dev_combined, lam_q, lam_m)
        acc = c / t if t else 0
        if acc > best_acc:
            best_lams, best_acc = (lam_q, lam_m), acc
    print(f"tuned on first half: lam_qwen={best_lams[0]:.2f} lam_mistral={best_lams[1]:.2f} "
          f"(tune acc {best_acc*100:.2f}%)")

    cat_correct, cat_total = {}, {}
    for i in report_idx:
        r, triples = dev_combined[i]
        cat = categorize(r["answer"])
        cat_total[cat] = cat_total.get(cat, 0) + 1
        pred = r["pred"] if triples is None else pick_best(triples, *best_lams)
        ok = (pred or "").strip().lower() == r["answer"].strip().lower()
        cat_correct[cat] = cat_correct.get(cat, 0) + ok

    print(f"\n--- FINAL dev accuracy (report half, frozen lambda, n={len(report_idx)}) ---")
    tot_c = tot_n = 0
    for cat in ("word", "symbol", "number"):
        c, t = cat_correct.get(cat, 0), cat_total.get(cat, 0)
        tot_c += c
        tot_n += t
        if t:
            print(f"{cat:8s} {c}/{t}  {c/t*100:.2f}%")
    if tot_n:
        print(f"{'overall':8s} {tot_c}/{tot_n}  {tot_c/tot_n*100:.2f}%")

    test_preds = []
    for r, triples in test_combined:
        pred = r["pred"] if triples is None else (pick_best(triples, *best_lams) or "")
        test_preds.append(pred)
    assert len(test_preds) == len(test_combined)
    with open(args.out_test_pred, "w", encoding="utf-8") as f:
        for p in test_preds:
            f.write(p + "\n")
    print(f"\nwrote {len(test_preds)} predictions -> {args.out_test_pred}")


if __name__ == "__main__":
    main()
