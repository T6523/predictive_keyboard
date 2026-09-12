"""Merges local's Stage 1 output (row-frac slice [0.0, 0.207)) with Kaggle's
(slice [0.207, 1.0)) into the full-length dev/test files mistral_only_infer.ipynb's
find_file("dev_qwen_scores.jsonl") expects. Sort-by-index + a contiguous-range check
catches a mismatched pair (wrong kernel output, stale local run, etc.) before it
silently produces a file with gaps or duplicates.

Usage:
    python3 merge_stage1_shards.py \
        --local-dev ../weights/stage1/dev_qwen_scores.jsonl \
        --local-test ../weights/stage1/test_qwen_scores.jsonl \
        --kaggle-dev ../weights/stage1_kaggle/dev_qwen_scores_kaggle.jsonl \
        --kaggle-test ../weights/stage1_kaggle/test_qwen_scores_kaggle.jsonl \
        --out-dir ../weights/stage1_merged
"""
import argparse
import json
import os


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def merge(local_rows, kaggle_rows, name):
    rows = local_rows + kaggle_rows
    rows.sort(key=lambda r: r["index"])
    idxs = [r["index"] for r in rows]
    assert idxs == list(range(len(idxs))), (
        f"{name}: merged indices aren't a contiguous 0..N-1 range -- "
        f"got {len(idxs)} rows spanning {idxs[0]}..{idxs[-1]} "
        "(overlap or gap between local/kaggle slices, or a stale/partial input file)")
    return rows


def demo():
    local = [{"index": 0, "v": "a"}, {"index": 1, "v": "b"}]
    kaggle = [{"index": 2, "v": "c"}, {"index": 3, "v": "d"}]
    merged = merge(local, kaggle, "demo")
    assert [r["v"] for r in merged] == ["a", "b", "c", "d"]
    try:
        merge(local, kaggle[1:], "demo")
        assert False, "should have raised on a gap"
    except AssertionError as e:
        assert "contiguous" in str(e)
    print("demo ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dev", default="../weights/stage1/dev_qwen_scores.jsonl")
    ap.add_argument("--local-test", default="../weights/stage1/test_qwen_scores.jsonl")
    ap.add_argument("--kaggle-dev", default="../weights/stage1_kaggle/dev_qwen_scores_kaggle.jsonl")
    ap.add_argument("--kaggle-test", default="../weights/stage1_kaggle/test_qwen_scores_kaggle.jsonl")
    ap.add_argument("--out-dir", default="../weights/stage1_merged")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for tag, local_path, kaggle_path in (
        ("dev", args.local_dev, args.kaggle_dev),
        ("test", args.local_test, args.kaggle_test),
    ):
        merged = merge(load_jsonl(local_path), load_jsonl(kaggle_path), tag)
        out_path = f"{args.out_dir}/{tag}_qwen_scores.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in merged:
                f.write(json.dumps(r) + "\n")
        print(f"{tag}: {len(merged)} rows -> {out_path}")


if __name__ == "__main__":
    main()
