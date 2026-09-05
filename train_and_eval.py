#!/usr/bin/env python3
"""Fine-tune unsloth/Qwen2.5-0.5B (LoRA) on the contest corpus, hard-stop at 8h, then run
constrained next-word eval on devv_eval.csv / devv_test.csv -- same task as scripts/eval.py and
transformer/, third approach: a pretrained subword LLM instead of a from-scratch word-level model.

Kaggle-only (needs a GPU + unsloth/trl/peft/bitsandbytes/transformers, none of which are in this
repo's local .venv -- same split as transformer/kernel/*, which also only runs on Kaggle).
Kaggle setup: enable internet (pip install + HF Hub download of the base model both need it --
the existing transformer/ kernel runs with internet off, this one can't) and, if the image
doesn't already have them:
    !pip install -q unsloth trl peft bitsandbytes

Reuses rather than rebuilds:
  - weights/vocab.txt for the candidate pool (already alnum-filtered by scripts/build_vocab.py --
    "parse train.src.tok to build the vocab" would just reproduce this file).
  - the symbol-letter rule from scripts/symbol_predict.py, inlined below (not imported --
    this file is pushed to Kaggle standalone, single-file, so it can't reach across the repo).

Real column names (checked against the actual csvs, not assumed): devv_eval.csv / devv_test.csv
have `context`, `first letter`, `answer` -- not history_text/prefix_char/target_next_word.

Usage:
    python3 train_and_eval.py
    python3 train_and_eval.py --eval-limit 500 --train-limit-blocks 200   # quick smoke test
"""
import argparse
import csv
import glob
import random
import subprocess
import sys
import time
from pathlib import Path

# bootstrap: Kaggle's stock image doesn't ship these -- installs once, on demand, so the script
# runs unmodified whether or not they're already present (no manual !pip install cell needed).
try:
    import unsloth  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth", "trl", "peft", "bitsandbytes"], check=True)

import numpy as np
import torch
from datasets import Dataset
from transformers import TrainerCallback, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

ROOT = Path(__file__).resolve().parent


# mirrors scripts/symbol_predict.py exactly (verified against dev_set/devv_eval/devv_test:
# 100% match, zero exceptions except '[' which always means [UNK]) -- keep the two in sync.
def predict_symbol(letter):
    return "[UNK]" if letter == "[" else letter


def is_symbol_letter(letter):
    return not str(letter).isalnum()
SEED = 42
MODEL_NAME = "unsloth/Qwen2.5-0.5B"
MAX_SEQ_LEN = 1024
TIME_BUDGET_SEC = 8 * 3600
OUT_DIR = ROOT / "weights" / "qwen_8hr_checkpoint"


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve(*rel_parts):
    """Kaggle mounts attached datasets under /kaggle/input/ -- notebook kernels flatten it to
    /kaggle/input/<slug>/, script kernels nest it as /kaggle/input/datasets/<user>/<slug>/
    (confirmed by running both) -- recursive glob covers whichever. Falls back to the repo path
    when running locally."""
    local = ROOT.joinpath(*rel_parts)
    if local.exists():
        return local
    hits = glob.glob(f"/kaggle/input/**/{rel_parts[-1]}", recursive=True)
    if hits:
        return Path(hits[0])
    raise FileNotFoundError(local)


# ---------------- vocab / prefix index ----------------
def load_vocab_by_letter(vocab_path):
    by_letter = {}
    with open(vocab_path, encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w and w.isalnum():
                by_letter.setdefault(w[0], []).append(w)
    return by_letter


# ---------------- training corpus: stream + pack, RAM-bounded ----------------
def stream_blocks(path, tokenizer, seq_len, eos_id):
    """One line at a time (3.8M lines / ~130M tokens) -- never holds more than the current
    partial block in memory, unlike a single batch_encode over the whole file."""
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            buf.extend(tokenizer.encode(line.strip(), add_special_tokens=False))
            buf.append(eos_id)
            while len(buf) > seq_len:
                yield buf[:seq_len]
                buf = buf[seq_len:]


def build_dataset(path, tokenizer, seq_len, limit_blocks=None):
    blocks = []
    for i, b in enumerate(stream_blocks(path, tokenizer, seq_len, tokenizer.eos_token_id)):
        blocks.append(b)
        if limit_blocks and i + 1 >= limit_blocks:
            break
    return Dataset.from_dict({"input_ids": blocks, "labels": [b[:] for b in blocks]})


# ---------------- 8-hour hard stop ----------------
class TimeBudgetCallback(TrainerCallback):
    def __init__(self, budget_sec):
        self.budget_sec = budget_sec
        self.start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.time()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - self.start >= self.budget_sec:
            print(f"time budget ({self.budget_sec}s) hit @ step {state.global_step}, stopping")
            control.should_training_stop = True
        return control


# ---------------- constrained scoring ----------------
def tokenize_vocab(tokenizer, vocab_by_letter):
    """word -> token id tuple, cached once (both eval files reuse it)."""
    words = [w for ws in vocab_by_letter.values() for w in ws]
    enc = tokenizer(words, add_special_tokens=False)["input_ids"]
    return {w: tuple(ids) for w, ids in zip(words, enc)}


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


@torch.no_grad()
def best_candidate(model, device, context_ids, candidates, max_batch=128):
    """candidates: list[(word, token_id_tuple)]. Picks argmax sum_i log P(tok_i | h, tok_<i>),
    teacher-forced. Grouped by shared token length and run as one batched forward per group
    (most words are 1 Qwen token -- only rare multi-token candidates cost a second/third group)
    instead of one forward per candidate."""
    by_len = {}
    for w, ids in candidates:
        by_len.setdefault(len(ids), []).append((w, ids))
    ctx_len = len(context_ids)
    best_w, best_s = None, float("-inf")
    for L, group in by_len.items():
        for chunk in batched(group, max_batch):
            words, id_lists = zip(*chunk)
            seqs = torch.tensor([context_ids + list(ids) for ids in id_lists], device=device)
            logits = model(seqs).logits
            logp = torch.log_softmax(logits[:, ctx_len - 1 : ctx_len - 1 + L, :].float(), dim=-1)
            target = torch.tensor(id_lists, device=device)
            score = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
            i = int(score.argmax())
            if score[i].item() > best_s:
                best_s, best_w = score[i].item(), words[i]
    return best_w


def run_eval(path, model, tokenizer, device, vocab_cache, vocab_by_letter, max_ctx, limit, out_csv):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    correct, out_rows = 0, []
    t0 = time.time()
    for n, row in enumerate(rows, 1):
        letter = row["first letter"]
        if is_symbol_letter(letter):
            pred = predict_symbol(letter)
        else:
            candidates = [(w, vocab_cache[w]) for w in vocab_by_letter.get(letter.lower(), ())]
            pred = None
            if candidates:
                ctx_ids = tokenizer.encode(row["context"], add_special_tokens=False)[-max_ctx:]
                pred = best_candidate(model, device, ctx_ids, candidates)
        is_correct = pred == row["answer"]
        correct += is_correct
        out_rows.append({"history": row["context"], "prefix": letter, "target": row["answer"],
                          "predicted": pred, "is_correct": is_correct})
        if n % 200 == 0:
            print(f"  {n}/{len(rows)} ({time.time() - t0:.0f}s)")
    acc = correct / max(len(rows), 1)
    print(f"{path.name}: accuracy {correct}/{len(rows)} = {acc:.4f}")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["history", "prefix", "target", "predicted", "is_correct"])
        w.writeheader()
        w.writerows(out_rows)
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--time-budget", type=float, default=TIME_BUDGET_SEC)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--eval-limit", type=int, default=None, help="cap rows scored per eval file (default: full set)")
    ap.add_argument("--train-limit-blocks", type=int, default=None, help="cap training blocks (quick smoke test)")
    args = ap.parse_args()

    seed_everything()

    print("/kaggle/input tree:", glob.glob("/kaggle/input/**/*", recursive=True))

    train_path = resolve("data", "train.src.tok")
    eval_path = resolve("data", "devv_eval.csv")
    test_path = resolve("data", "devv_test.csv")
    vocab_path = resolve("weights", "vocab.txt")

    # ponytail: T4 (this kernel's GPU) is pre-Ampere -- no bf16 support, confirmed by unsloth's own
    # "Device does not support bfloat16. Will change to float16." log line. Picking the dtype from
    # torch.cuda.is_bf16_supported() instead of hardcoding bf16 keeps this working on both T4 and
    # any future Ampere+ machine_shape.
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

    print("--- loading base model + LoRA ---")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, dtype=compute_dtype, load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=SEED,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("--- packing training corpus (streamed, line-by-line) ---")
    t0 = time.time()
    train_ds = build_dataset(train_path, tokenizer, MAX_SEQ_LEN, limit_blocks=args.train_limit_blocks)
    print(f"{len(train_ds)} blocks of {MAX_SEQ_LEN} tokens ({time.time() - t0:.0f}s)")

    training_args = TrainingArguments(
        output_dir=str(args.out / "trainer_tmp"),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_8bit",
        weight_decay=0.01,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        logging_steps=20,
        save_strategy="no",  # TimeBudgetCallback + the explicit save_pretrained below handle persistence
        report_to="none",
        seed=SEED,
    )

    # ponytail: dataset already has input_ids/labels (pre-tokenized above), so no dataset_text_field/
    # formatting_func is passed -- SFTTrainer is expected to skip its own tokenization when those
    # columns are already present. Untested against the exact trl version Kaggle ships (no local
    # env here to pin against); if it errors on this, the fallback is a plain
    # transformers.Trainer + DataCollatorForLanguageModeling(tokenizer, mlm=False), which does the
    # identical causal-LM step without trl's dataset-format assumptions.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=training_args,
        callbacks=[TimeBudgetCallback(args.time_budget)],
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
    )

    print("--- training ---")
    trainer.train()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"saved -> {args.out}")

    # loss curve: trainer.state.log_history already has one {"loss": ..., "step": ...} dict per
    # logging_steps interval -- no separate tracking needed, just plot what's already collected.
    steps = [e["step"] for e in trainer.state.log_history if "loss" in e]
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    if steps:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(steps, losses)
        plt.xlabel("step"); plt.ylabel("train loss"); plt.title("Qwen2.5-0.5B LoRA fine-tune")
        plt.tight_layout()
        plt.savefig(args.out / "loss_curve.png")
        plt.close()
        print(f"saved -> {args.out / 'loss_curve.png'} ({len(steps)} points)")

    print("--- constrained eval ---")
    FastLanguageModel.for_inference(model)
    vocab_by_letter = load_vocab_by_letter(vocab_path)
    vocab_cache = tokenize_vocab(tokenizer, vocab_by_letter)
    max_ctx = MAX_SEQ_LEN - 8  # room for the longest candidate's tokens

    eval_acc = run_eval(eval_path, model, tokenizer, device, vocab_cache, vocab_by_letter, max_ctx,
                         args.eval_limit, ROOT / "devv_eval_predictions.csv")
    test_acc = run_eval(test_path, model, tokenizer, device, vocab_cache, vocab_by_letter, max_ctx,
                         args.eval_limit, ROOT / "devv_test_predictions.csv")
    print(f"devv_eval accuracy: {eval_acc:.4f}")
    print(f"devv_test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
