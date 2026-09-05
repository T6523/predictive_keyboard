#!/usr/bin/env python3
"""Part 1: train two KenLM n-gram models (lmplz + build_binary trie), score dev/test csvs
with both, save everything under a fresh weights/run_<timestamp>/ folder.

Model A: data/train.src.tok
Model B: data/gigaword.tok  (assumed already vocab-masked to match Model A)

Usage:
    python3 pipeline.py
    python3 pipeline.py --order 5 --memory 12G --prune 0 0 1
    python3 pipeline.py --self-test     # tiny end-to-end smoke test, no real data needed

Requires scripts/bin/{lmplz,build_binary} (build with scripts/build_kenlm.sh) and the
`kenlm` python package (pip install https://github.com/kpu/kenlm/archive/master.zip).

Reproducibility: lmplz's external-memory count/sort/estimate algorithm has no thread-count
or seed knob in this build -- same corpus bytes + same -o/-S/--prune/--discount_fallback
always produce the same ARPA, so build_binary's trie and every downstream score are byte-
identical run to run. Only the run_dir name (timestamped, one per invocation) varies.

RAM during inference: score_csv streams the csv row-by-row (csv.DictReader/DictWriter, no
pandas/full-file load) and both models are memory-mapped tries (kenlm.Model mmaps the
.klm), so RAM stays flat regardless of csv or corpus size.
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_BIN = ROOT / "scripts" / "bin"


def train_klm(corpus, out_dir, name, lmplz_bin, build_binary_bin, order, memory, prune, tmp_dir):
    """corpus -> lmplz -> ARPA -> build_binary trie -> .klm. ARPA deleted after. Returns klm path."""
    arpa_path = out_dir / f"{name}.arpa"
    klm_path = out_dir / f"{name}.klm"
    log_path = out_dir / f"{name}.lmplz.log"

    t0 = time.time()
    with open(corpus, "rb") as fin, open(arpa_path, "wb") as fout, open(log_path, "w") as flog:
        subprocess.run(
            [str(lmplz_bin), "-o", str(order), "-S", memory, "-T", str(tmp_dir),
             "--prune", *prune, "--discount_fallback"],
            stdin=fin, stdout=fout, stderr=flog, check=True,
        )
    print(f"  lmplz [{name}]: {time.time()-t0:.0f}s -> {arpa_path.name} (log: {log_path.name})")

    t0 = time.time()
    subprocess.run([str(build_binary_bin), "trie", str(arpa_path), str(klm_path)], check=True)
    arpa_path.unlink()  # spec: delete ARPA after compiling to binary
    print(f"  build_binary [{name}]: {time.time()-t0:.0f}s -> {klm_path.name} (arpa deleted)")
    return klm_path


def score_csv(model_a, model_b, in_csv, out_csv, text_col):
    """Stream in_csv -> out_csv, appending model_a_logprob10 / model_b_logprob10 columns."""
    with open(in_csv, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames + ["model_a_logprob10", "model_b_logprob10"]
        with open(out_csv, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            n = 0
            for row in reader:
                text = row[text_col]
                row["model_a_logprob10"] = model_a.score(text)
                row["model_b_logprob10"] = model_b.score(text)
                writer.writerow(row)
                n += 1
    return n


def run(args):
    import kenlm  # deferred: not needed for --help, and self-test builds tiny models first

    run_dir = ROOT / "weights" / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_dir}")

    with tempfile.TemporaryDirectory(dir=run_dir) as tmp_dir:
        print("training model A...")
        klm_a = train_klm(args.train_a, run_dir, "model_a", args.lmplz, args.build_binary,
                           args.order, args.memory, args.prune, tmp_dir)
        print("training model B...")
        klm_b = train_klm(args.train_b, run_dir, "model_b", args.lmplz, args.build_binary,
                           args.order, args.memory, args.prune, tmp_dir)

    print("loading models for inference...")
    model_a = kenlm.Model(str(klm_a))
    model_b = kenlm.Model(str(klm_b))

    for in_csv, out_name in [(args.eval_csv, "devv_eval_predictions.csv"),
                              (args.test_csv, "devv_test_predictions.csv")]:
        out_csv = run_dir / out_name
        n = score_csv(model_a, model_b, in_csv, out_csv, args.text_col)
        print(f"scored {n} rows from {in_csv} -> {out_csv}")

    print(f"done -> {run_dir}")
    return run_dir


def self_test():
    """Tiny end-to-end smoke test: fake corpora + csv, order-2 model, real lmplz/build_binary/kenlm."""
    import kenlm

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        corpus_a = td / "a.tok"
        corpus_b = td / "b.tok"
        corpus_a.write_text("the cat sat on the mat\nthe dog sat on the log\n")
        corpus_b.write_text("a cat ran in the park\na dog ran in the yard\n")

        (td / "weights").mkdir()
        run_dir = td / "weights" / "run_test"
        run_dir.mkdir()
        with tempfile.TemporaryDirectory(dir=run_dir) as tmp_dir:
            klm_a = train_klm(corpus_a, run_dir, "model_a", DEFAULT_BIN / "lmplz",
                               DEFAULT_BIN / "build_binary", order=2, memory="1G",
                               prune=["0"], tmp_dir=tmp_dir)
            klm_b = train_klm(corpus_b, run_dir, "model_b", DEFAULT_BIN / "lmplz",
                               DEFAULT_BIN / "build_binary", order=2, memory="1G",
                               prune=["0"], tmp_dir=tmp_dir)
        assert klm_a.exists() and klm_b.exists()
        assert not (run_dir / "model_a.arpa").exists(), "arpa should be deleted"

        model_a = kenlm.Model(str(klm_a))
        model_b = kenlm.Model(str(klm_b))

        in_csv = td / "dev.csv"
        with open(in_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["context", "first letter", "answer"])
            w.writerow(["the cat sat on the", "m", "mat"])

        out_csv = td / "out.csv"
        n = score_csv(model_a, model_b, in_csv, out_csv, "context")
        assert n == 1
        with open(out_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        row = rows[0]
        assert set(row) == {"context", "first letter", "answer", "model_a_logprob10", "model_b_logprob10"}
        assert float(row["model_a_logprob10"]) > float(row["model_b_logprob10"]), (
            "model A saw this exact sentence, should score it higher than model B"
        )
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-a", default=ROOT / "data" / "train.src.tok", type=Path)
    ap.add_argument("--train-b", default=ROOT / "data" / "gigaword.tok", type=Path)
    ap.add_argument("--eval-csv", default=ROOT / "data" / "devv_eval.csv", type=Path)
    ap.add_argument("--test-csv", default=ROOT / "data" / "devv_test.csv", type=Path)
    ap.add_argument("--text-col", default="context", help="csv column to score")
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--memory", default="12G", help="lmplz -S RAM budget")
    ap.add_argument("--prune", default="0 0 1", help="lmplz --prune thresholds, space-separated")
    ap.add_argument("--lmplz", default=DEFAULT_BIN / "lmplz", type=Path)
    ap.add_argument("--build-binary", default=DEFAULT_BIN / "build_binary", type=Path)
    ap.add_argument("--self-test", action="store_true", help="run tiny smoke test instead")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    for exe in (args.lmplz, args.build_binary):
        if not exe.exists():
            sys.exit(f"missing {exe} -- build with scripts/build_kenlm.sh first")
    args.prune = args.prune.split()

    run(args)


if __name__ == "__main__":
    main()
