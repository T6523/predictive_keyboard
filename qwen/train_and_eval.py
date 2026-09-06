#!/usr/bin/env python3
"""Fine-tune unsloth/Qwen2.5-0.5B (LoRA) on the contest corpus, hard-stop at 8h, then run
constrained next-word eval on devv_eval.csv / devv_test.csv -- same task as scripts/eval.py and
gpt/, third approach: a pretrained subword LLM instead of a from-scratch word-level model.

Kaggle-only (needs a GPU + unsloth/trl/peft/bitsandbytes/transformers, none of which are in this
repo's local .venv -- same split as gpt/kernel/*, which also only runs on Kaggle).
Kaggle setup: enable internet (pip install + HF Hub download of the base model both need it --
the existing gpt/ kernel runs with internet off, this one can't) and, if the image
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
    python3 qwen/train_and_eval.py
    python3 qwen/train_and_eval.py --eval-limit 500 --train-limit-blocks 200   # quick smoke test
"""
import argparse
import copy
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
try:
    from peft.import_utils import is_torchao_available
    is_torchao_available()  # raises if the image's preinstalled torchao is too old for this peft
except ImportError:
    # Kaggle's stock image shipped torchao 0.10.0, but peft's LoRA dispatcher unconditionally
    # requires >=0.16.0 (checked even though our adapter has nothing to do with torchao) --
    # confirmed via a real run's traceback in eval_kernel/eval_qwen.py (same bootstrap issue).
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "peft", "torchao"], check=True)

import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from unsloth import FastLanguageModel

ROOT = Path(__file__).resolve().parent.parent
# ponytail: Kaggle script kernels mount the code dir (ROOT, /kaggle/src) read-only -- only
# /kaggle/working is writable. Falls back to ROOT when running locally (no /kaggle/working there).
WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else ROOT


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
OUT_DIR = WORK_DIR / "weights" / "qwen_8hr_checkpoint"


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


def _batch_repeat_past(past, n):
    """Version-robust past_key_values repeat -- see eval_kernel/eval_qwen.py's copy of this
    function for why (newer transformers' DynamicCache.batch_repeat_interleave mutates in place
    and returns None; Kaggle's shipped 5.5.0 instead returns the legacy tuple-of-(key,value)
    format, which has no such method at all)."""
    if hasattr(past, "batch_repeat_interleave"):
        past.batch_repeat_interleave(n)
        return past
    return tuple((k.repeat_interleave(n, dim=0), v.repeat_interleave(n, dim=0)) for k, v in past)


@torch.no_grad()
def best_candidate(model, device, context_ids, candidates, max_batch=64):
    """candidates: list[(word, token_id_tuple)]. Picks argmax sum_i log P(tok_i | h, tok_<i>),
    teacher-forced. Grouped by shared token length; context is encoded ONCE (one forward) and its
    past_key_values reused across every candidate batch, so each batch only pays for its own 1-6
    new tokens instead of re-running the full context through every layer per batch.

    Was originally a per-batch full recompute gated by logits_to_keep=L (the OOM fix: an 8hr run
    hit model(seqs).logits materializing the FULL (batch, ctx_len+L, vocab) tensor regardless of
    L, and Qwen's ~152k vocab made that ~37GB of logits alone at batch=128, past a T4's 14.56GB).
    That version also had a real correctness bug -- logits_to_keep=L returns the model's LAST L
    logit rows, which predict tokens L+1..2L relative to context end, not tokens 1..L, so scoring
    target=candidate directly against those rows was off by one position (verified empirically,
    including the L=1 case). Replaced wholesale with the KV-cache version below, kept in sync with
    eval_kernel/eval_qwen.py's best_candidate() -- see that file's module docstring for the
    other bug this caught (DynamicCache.batch_repeat_interleave mutates in place, returns None)."""
    ctx = torch.tensor([context_ids], device=device)
    out = model(ctx, use_cache=True)
    base_past = out.past_key_values
    ctx_last_logp = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
    ctx_len = len(context_ids)

    by_len = {}
    for w, ids in candidates:
        by_len.setdefault(len(ids), []).append((w, ids))

    best_w, best_s = None, float("-inf")
    for L, group in by_len.items():
        for chunk in batched(group, max_batch):
            words, id_lists = zip(*chunk)
            first_term = torch.stack([ctx_last_logp[ids[0]] for ids in id_lists])
            if L == 1:
                score = first_term
            else:
                batch_past = _batch_repeat_past(copy.deepcopy(base_past), len(chunk))
                target = torch.tensor(id_lists, device=device)
                cache_position = torch.arange(ctx_len, ctx_len + L, device=device)
                logits = model(target, past_key_values=batch_past, cache_position=cache_position,
                                use_cache=False).logits
                logp = torch.log_softmax(logits.float(), dim=-1)
                # logp[:, j] predicts target[:, j+1] (j=0..L-2); target[:, 0]'s own probability is
                # first_term, already scored off the context pass.
                rest = logp[:, :-1, :].gather(-1, target[:, 1:].unsqueeze(-1)).squeeze(-1).sum(dim=-1)
                score = first_term + rest
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

    # ponytail: dataset already has input_ids/labels (pre-tokenized above) so SFTTrainer buys nothing
    # here -- it exists to turn raw text into that pair, we already did it. Plain Trainer +
    # DataCollatorForLanguageModeling(mlm=False) does the identical causal-LM step without chasing
    # trl's SFTTrainer/SFTConfig API churn (v5-era trl renamed tokenizer->processing_class and moved
    # max_seq_length/packing into SFTConfig -- confirmed broken on Kaggle's shipped version).
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        args=training_args,
        callbacks=[TimeBudgetCallback(args.time_budget)],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
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
    # ponytail: reload plain (no unsloth) instead of FastLanguageModel.for_inference(model) on the
    # live training model -- unsloth's patched forward routes any past_key_values-is-not-None call
    # through its single-token generate() decode path (hard-asserts q_len==1), incompatible with
    # best_candidate()'s multi-token KV-cache batches. See eval_kernel/eval_qwen.py's module
    # docstring (bug #3) for the traceback and the merge_and_unload() fix, reused here verbatim --
    # args.out already has the adapter this training run just saved two lines up.
    del model
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=compute_dtype)
    model = PeftModel.from_pretrained(base, str(args.out)).merge_and_unload()
    model.eval().to(device)

    vocab_by_letter = load_vocab_by_letter(vocab_path)
    vocab_cache = tokenize_vocab(tokenizer, vocab_by_letter)
    max_ctx = MAX_SEQ_LEN - 8  # room for the longest candidate's tokens

    eval_acc = run_eval(eval_path, model, tokenizer, device, vocab_cache, vocab_by_letter, max_ctx,
                         args.eval_limit, WORK_DIR / "devv_eval_predictions.csv")
    test_acc = run_eval(test_path, model, tokenizer, device, vocab_cache, vocab_by_letter, max_ctx,
                         args.eval_limit, WORK_DIR / "devv_test_predictions.csv")
    print(f"devv_eval accuracy: {eval_acc:.4f}")
    print(f"devv_test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
