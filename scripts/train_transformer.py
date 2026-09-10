"""Plain causal-LM QLoRA fine-tune on train_final.src.tok (TODO.md Suggestion item 2):
no more per-sentence context+"[letter]"->target objective -- every position in every
packed sequence is now a supervised target, ~30x more supervision per token processed
vs the old one-target-per-sentence scheme. The first-letter constraint moves entirely
to inference (infer_transformer.py's logit mask over `boundary + letter`-prefixed
vocab pieces); baking "[letter]" into the training prompt taught the model nothing the
mask doesn't already provide for free, and cost ~30x sample efficiency for it.

Uses Unsloth's FastLanguageModel (2-5x faster, ~50% less VRAM) + TRL's SFTTrainer with
packing=True: many short lines get packed into one training sequence instead of each
paying full padding, so p99 line length (~57 tok, see TODO.md Data statistics) barely
matters -- packing amortizes it away.

Usage:
    python3 train_transformer.py --train ../data/train_final.src.tok \
        --out ../weights/qwen3b_lora --max-steps 2000 --max-lines 2000000

    # 6h Kaggle chunk, second session picking up where session 1 left off:
    python3 train_transformer.py --no-4bit --max-hours 6 --skip-lines 900000 \
        --out ../weights/qwen3b_lora --resume ../weights/qwen3b_lora

Requires: torch, transformers, trl, peft, bitsandbytes, unsloth (already in .venv).
"""
import argparse
import time

import torch

MODEL_ID = "Qwen/Qwen2.5-3B"


def load_lines(path, max_lines, skip=0):
    """Read up to max_lines non-empty lines starting after skip raw lines (0 = whole
    file). skip lets a later Kaggle session train on the next chunk of the corpus
    instead of repeating the first one -- see --skip-lines / --max-hours below.
    """
    with open(path, encoding="utf-8") as f:
        for _ in range(skip):
            next(f, None)
        if max_lines:
            lines = [next(f, "").strip() for _ in range(max_lines)]
        else:
            lines = [line.strip() for line in f]
    return [line for line in lines if line]


def demo():
    """Self-check: line loading + empty-line filtering, no model/network needed."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("the quick brown fox\n\nsolo\n")
        path = f.name
    lines = load_lines(path, 0)
    skipped = load_lines(path, 0, skip=1)
    os.unlink(path)
    assert lines == ["the quick brown fox", "solo"]
    assert skipped == ["solo"]
    print("demo ok:", lines, skipped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--train", default="../data/train_final.src.tok")
    ap.add_argument("--out", default="../weights/qwen3b_lora")
    ap.add_argument("--skip-lines", type=int, default=0, help="skip this many raw lines before reading -- point at the next chunk for a follow-up session (see --max-hours)")
    ap.add_argument("--max-lines", type=int, default=900_000, help="0 = rest of the 3.8M-line file. Default is a buffer for one ~6h chunk at measured throughput (~1200 tok/s local) -- --max-hours is what actually stops the run")
    ap.add_argument("--max-hours", type=float, default=0, help="stop after this many wall-clock hours regardless of --max-steps (0 = disabled, step-count only) -- use this to fit a Kaggle session")
    ap.add_argument("--max-steps", type=int, default=100_000, help="hard upper bound; --max-hours is the real stopper for a timed session")
    ap.add_argument("--resume", default=None, help="LoRA adapter dir to resume from (e.g. previous chunk's --out) instead of the base model's fresh adapter")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-length", type=int, default=512, help="packed sequence length")
    ap.add_argument("--no-4bit", action="store_true", help="bf16 LoRA instead of QLoRA (more VRAM, more throughput -- use on Kaggle T4x2)")
    ap.add_argument("--demo", action="store_true", help="run self-check and exit")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    from unsloth import FastLanguageModel  # must import before trl/transformers (Unsloth's own requirement --
    from datasets import Dataset            # importing it later applies its monkeypatches incorrectly and
    from trl import SFTConfig, SFTTrainer   # corrupts SFTConfig.eos_token via a to_dict() round-trip bug)
    from transformers import TrainerCallback

    class TimeLimit(TrainerCallback):
        """ponytail: wall-clock stop, not a step-count guess -- packed-sequence count
        per session is hard to predict exactly, wall time isn't."""
        def __init__(self, max_hours):
            self.deadline = time.time() + max_hours * 3600 if max_hours else None

        def on_step_end(self, args, state, control, **kwargs):
            if self.deadline and time.time() >= self.deadline:
                control.should_training_stop = True
            return control

    model, tokenizer = FastLanguageModel.from_pretrained(
        args.resume or args.model, max_seq_length=args.max_seq_length, load_in_4bit=not args.no_4bit,
    )
    if not args.resume:
        model = FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    lines = load_lines(args.train, args.max_lines, args.skip_lines)
    ds = Dataset.from_dict({"text": lines})

    # T4 (Kaggle's free GPU) has no fast bf16 tensor cores (Turing, needs Ampere+) --
    # bf16=True there silently runs slow/emulated. Detect instead of assuming the 4060's bf16.
    bf16_ok = torch.cuda.is_bf16_supported()
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=args.out,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.max_steps,
            max_length=args.max_seq_length,
            packing=True,
            dataset_text_field="text",
            bf16=bf16_ok,
            fp16=not bf16_ok,
            logging_steps=20,
            save_steps=500,
            save_total_limit=2,
            report_to=[],
        ),
        callbacks=[TimeLimit(args.max_hours)],
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    return trainer.state.log_history


if __name__ == "__main__":
    main()
