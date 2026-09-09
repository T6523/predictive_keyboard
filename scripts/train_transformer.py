"""QLoRA fine-tune mistralai/Mistral-7B-v0.1 on train_final.src.tok for the predictive
keyboard task: given left context + first letter of next word, predict the word.

Builds training examples on the fly from raw sentences (no separate context/answer CSV
needed for train) -- for each sentence, pick a random split point i, use tokens[:i] as
context and tokens[i] as target, first letter = target[0]. Loss is masked to the target
word + EOS only (prompt tokens don't contribute), matching what dev/test actually score.

Usage:
    python3 train_transformer.py --train ../data/train_final.src.tok \
        --out ../weights/mistral7b_lora --max-steps 2000 --max-lines 2000000

Requires: torch, transformers, peft, bitsandbytes, accelerate (already in .venv).
"""
import argparse
import random

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "mistralai/Mistral-7B-v0.1"


def make_example(line, rng):
    """One sentence -> (context_str, first_letter, target_word). None if unsplittable.

    Split point is never the first token (need >=1 token of context) or the sentence's
    only token.
    """
    toks = line.split()
    if len(toks) < 2:
        return None
    i = rng.randint(1, len(toks) - 1)
    return " ".join(toks[:i]), toks[i][0], toks[i]


def build_features(example, tokenizer, max_len=256):
    """Tokenize prompt+target, mask prompt tokens to -100 so loss is target-word-only."""
    context, letter, target = example
    prompt = f"{context} [{letter}]"
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    target_ids = tokenizer(" " + target, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + target_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + target_ids)[:max_len]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


class SentenceDataset(torch.utils.data.Dataset):
    """Lazily samples a random split per __getitem__ -- same sentence yields different
    (context, target) pairs across epochs, cheap way to multiply examples out of 3.8M lines.
    """

    def __init__(self, path, tokenizer, max_lines, max_len, seed=0):
        with open(path, encoding="utf-8") as f:
            self.lines = [next(f) for _ in range(max_lines)] if max_lines else f.readlines()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        ex = None
        while ex is None:  # skip degenerate (single-token) lines
            ex = make_example(self.lines[idx].strip(), self.rng)
            if ex is None:
                idx = self.rng.randrange(len(self.lines))
        return build_features(ex, self.tokenizer, self.max_len)


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append(b["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def demo():
    """Self-check: split logic + masking, no model/network needed."""
    rng = random.Random(42)
    ex = make_example("the quick brown fox jumps", rng)
    assert ex is not None
    context, letter, target = ex
    assert context.split()[-1] != target  # target excluded from context
    assert letter == target[0]
    assert make_example("solo", rng) is None  # single-token line -> unsplittable
    print("demo ok:", ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/train_final.src.tok")
    ap.add_argument("--out", default="../weights/mistral7b_lora")
    ap.add_argument("--max-lines", type=int, default=2_000_000, help="0 = full 3.8M-line file")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--demo", action="store_true", help="run self-check and exit")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    model.print_trainable_parameters()

    ds = SentenceDataset(args.train, tokenizer, args.max_lines, args.max_len)

    trainer = Trainer(
        model=model,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
        args=TrainingArguments(
            output_dir=args.out,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.max_steps,
            bf16=True,
            logging_steps=20,
            save_steps=500,
            save_total_limit=2,
            report_to=[],
        ),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)


if __name__ == "__main__":
    main()
