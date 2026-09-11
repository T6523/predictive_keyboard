"""Generates qwen3b_matched_eval.ipynb. Run once locally, not on Kaggle -- ponytail:
building the ipynb JSON by hand in an editor is error-prone, a generator script is the
lazy-but-correct path, same convention as the training notebook's now-expired one."""
import json

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": src.splitlines(keepends=True)})


md("""# Qwen2.5-3B matched zero-shot vs fine-tuned comparison + blend-ready score export

Runs the SAME stratified 10% dev sample (seed 42, word-category only -- symbol is
deterministic and number now routes entirely to the local n-gram, see report.md's
routing scheme) through beam search (k=5) twice: base Qwen2.5-3B (zero-shot) and the
same base + the trained LoRA adapter. Apples-to-apples this time (same rows, same
beam-k5 protocol, unlike the greedy-vs-beam comparison used before).

Exports each row's top-5 candidates **with sequence log-prob scores** (not just
ranked words) so the n-gram blend (`λ·logP_lm + (1-λ)·logP_ngram`) can happen locally
-- kenlm/the custom n-gram counts model only exist locally, this notebook's job is
just to hand back scored LM candidates.

**Inputs**: dataset `predictive-keyboard` (dev_set_final.csv), model `qwen2.5`
(transformers/3b), dataset `qwen3b-lora-ckpt` (trained adapter).

**Output**: `/kaggle/working/qwen_matched_scores.csv` -- context, first_letter,
answer, category, zeroshot_preds (`word:score; word:score; ...`), lora_preds (same
format). Plain transformers, no unsloth needed for inference-only.""")

code('!pip install -q -U transformers peft bitsandbytes accelerate')

code("""import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    LogitsProcessor, StoppingCriteria,
)
from peft import PeftModel
import csv, re, random, time, glob, os, json""")

code("""print("/kaggle/input tree:")
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.count(os.sep) - "/kaggle/input".count(os.sep)
    if depth <= 3:
        print("  " * depth + os.path.basename(root) + "/")

def find_file(name):
    matches = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert matches, f"{name} not found under /kaggle/input -- check the dataset is attached"
    return matches[0]

def find_qwen_base():
    # content-based find, not a guessed mount path -- see training notebook's note
    for cfg_path in glob.glob("/kaggle/input/**/config.json", recursive=True):
        try:
            cfg = json.load(open(cfg_path))
        except (json.JSONDecodeError, OSError):
            continue
        if cfg.get("model_type") == "qwen2":
            return os.path.dirname(cfg_path)
    return None

def find_lora_adapter():
    matches = glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
    assert matches, "qwen3b-lora-ckpt dataset not attached"
    return os.path.dirname(matches[0])

DEV_PATH = find_file("dev_set_final.csv")
MODEL_PATH = find_qwen_base()
LORA_PATH = find_lora_adapter()
assert MODEL_PATH, "Qwen2.5 base model not found under /kaggle/input"
print("DEV_PATH:", DEV_PATH)
print("MODEL_PATH:", MODEL_PATH)
print("LORA_PATH:", LORA_PATH)""")

code("""CAT_NUMBER = re.compile(r"[0-9]+")
CAT_WORD = re.compile(r"(?=.*[a-z])[a-z']+", re.IGNORECASE)  # >=1 real letter -- a bare "'" isn't a word

def categorize(tok):
    if CAT_NUMBER.fullmatch(tok):
        return "number"
    if CAT_WORD.fullmatch(tok):
        return "word"
    return "symbol"

def detect_boundary(tokenizer):
    vocab = tokenizer.get_vocab()
    counts = {"\\u2581": 0, "\\u0120": 0}
    for piece in vocab:
        if piece[:1] in counts:
            counts[piece[:1]] += 1
    return max(counts, key=counts.get)

def build_letter_masks(tokenizer, boundary, device, vocab_size):
    vocab = tokenizer.get_vocab()
    letters = list("abcdefghijklmnopqrstuvwxyz0123456789")
    masks = {c: torch.zeros(vocab_size, dtype=torch.bool) for c in letters}
    for piece, idx in vocab.items():
        if len(piece) > 1 and piece[0] == boundary and piece[1].lower() in masks:
            masks[piece[1].lower()][idx] = True
    return {c: m.to(device) for c, m in masks.items()}

def get_boundary_ids(tokenizer, boundary):
    return [idx for piece, idx in tokenizer.get_vocab().items() if piece.startswith(boundary)]

class FirstTokenLetterMask(LogitsProcessor):
    def __init__(self, letter_mask, prompt_len):
        self.letter_mask = letter_mask
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] == self.prompt_len:
            scores = scores.masked_fill(~self.letter_mask, float("-inf"))
        return scores

class StopAtWordBoundary(StoppingCriteria):
    def __init__(self, prompt_len, boundary_ids_tensor, eos_id):
        self.prompt_len = prompt_len
        self.boundary_ids_tensor = boundary_ids_tensor
        self.eos_id = eos_id

    def __call__(self, input_ids, scores, **kwargs):
        cur_len = input_ids.shape[1]
        if cur_len <= self.prompt_len + 1:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        last = input_ids[:, -1]
        return torch.isin(last, self.boundary_ids_tensor) | (last == self.eos_id)""")

code('''@torch.inference_mode()
def predict_topk_scored(model, tokenizer, contexts, letters, masks, boundary_ids_tensor, device, k, max_extra=4):
    """Same beam-search machinery as scripts/infer_topk.py's predict_topk_batch, plus
    each kept candidate's sequence log-prob score (length-normalized by generate()'s
    own length_penalty) -- needed for the local n-gram blend, not just ranking."""
    enc = tokenizer(contexts, return_tensors="pt", padding=True).to(device)
    letter_mask = torch.stack([masks[l.lower()] for l in letters]).repeat_interleave(k, dim=0)
    prompt_len = enc["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id

    out = model.generate(
        **enc, max_new_tokens=max_extra + 1, num_beams=k, num_return_sequences=k,
        do_sample=False, early_stopping=True,
        logits_processor=[FirstTokenLetterMask(letter_mask, prompt_len)],
        stopping_criteria=[StopAtWordBoundary(prompt_len, boundary_ids_tensor, eos_id)],
        pad_token_id=eos_id, return_dict_in_generate=True, output_scores=True,
        # output_scores=True is required for generate() to populate sequences_scores at
        # all in this transformers version -- return_dict_in_generate=True alone left it
        # None (hit this on the first Kaggle run, AttributeError on .tolist()).
    )
    seq_scores = out.sequences_scores.tolist()
    generated = out.sequences[:, prompt_len:].tolist()

    boundary_set = set(boundary_ids_tensor.tolist())
    results = []
    for i in range(0, len(generated), k):
        seen, ranked = set(), []
        for row, sc in zip(generated[i:i + k], seq_scores[i:i + k]):
            cut = next((j for j, tid in enumerate(row) if j > 0 and (tid in boundary_set or tid == eos_id)), len(row))
            word = tokenizer.decode(row[:cut]).strip()
            key = word.lower()
            if key not in seen:
                seen.add(key)
                ranked.append((word, sc))
        results.append(ranked)
    return results''')

code("""tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
device = "cuda" if torch.cuda.is_available() else "cpu"
boundary = detect_boundary(tokenizer)

with open(DEV_PATH, encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f))
print(f"{len(all_rows)} dev rows total")

# Same stratified 10% sample as infer_topk.py / the training notebook's inference cell --
# same seed, same rows, so this stays comparable to every number already reported.
SAMPLE_FRAC = 0.10
random.seed(42)
by_cat = {}
for r in all_rows:
    by_cat.setdefault(categorize(r["answer"]), []).append(r)
sample = []
for cat, group in by_cat.items():
    n = max(1, round(len(group) * SAMPLE_FRAC))
    sample.extend(random.sample(group, n))
print(f"stratified {SAMPLE_FRAC*100:.0f}% sample: {len(sample)} rows "
      f"(source sizes: {{c: len(g) for c, g in by_cat.items()}})")

# Only word-category rows go through the model this run -- symbol is deterministic
# (letter copy) and number now routes entirely to the local n-gram (see report.md's
# routing scheme), so there's nothing for either zero-shot or fine-tuned to add there.
word_rows = [r for r in sample if categorize(r["answer"]) == "word"]
other_rows = [r for r in sample if categorize(r["answer"]) != "word"]
print(f"{len(word_rows)} word rows to run through both models, {len(other_rows)} other rows passed through untouched")""")

code("""BATCH_SIZE = 12  # T4 16GB -- beam search (k=5) multiplies VRAM ~5x per row, keep modest
K = 5

def run_pass(model, label):
    preds = {}
    vocab_size = model.get_output_embeddings().weight.shape[0]
    masks = build_letter_masks(tokenizer, boundary, device, vocab_size)
    boundary_ids_tensor = torch.tensor(get_boundary_ids(tokenizer, boundary), device=device)
    model.generation_config.max_length = None
    t0 = time.time()
    for start in range(0, len(word_rows), BATCH_SIZE):
        batch = word_rows[start:start + BATCH_SIZE]
        contexts = [r["context"] for r in batch]
        letters = [r["first letter"] for r in batch]
        ranked = predict_topk_scored(model, tokenizer, contexts, letters, masks, boundary_ids_tensor, device, K)
        for r, cands in zip(batch, ranked):
            preds[id(r)] = cands
        done = start + len(batch)
        if done % (BATCH_SIZE * 10) == 0 or done == len(word_rows):
            elapsed = time.time() - t0
            print(f"[{label}] [{done}/{len(word_rows)}] {elapsed:.1f}s, {done/max(elapsed,1e-9):.2f} rows/s", flush=True)
    return preds""")

code("""bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
# device_map="auto" on this Kaggle T4x2 shape shards the model across both GPUs --
# fine for raw throughput-per-flop, catastrophic for autoregressive generation (every
# layer boundary crossing GPUs is a sync point, hit every token x every beam). The 3B
# 4bit model fits in one T4's 16GB easily -- pin to one GPU instead (measured 1.49
# rows/s with device_map="auto" vs local's 7.6 rows/s at the same k/batch -- a 5x gap
# too big to be T4-vs-4060 alone; this was the fix, see conversation).
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map={"": 0})
model.eval()

zeroshot_preds = run_pass(model, "zero-shot")""")

code("""model = PeftModel.from_pretrained(model, LORA_PATH)  # adapter active on the same
                                                       # quantized base -- one model load,
                                                       # not two, saves the reload time/VRAM
model.eval()

lora_preds = run_pass(model, "lora")""")

code("""def acc_table(preds, label):
    tot = {1: 0, 5: 0}
    n = len(word_rows)
    for r in word_rows:
        ans = r["answer"].strip().lower()
        cands = [w.strip().lower() for w, _ in preds[id(r)]]
        if cands and cands[0] == ans:
            tot[1] += 1
        if ans in cands[:5]:
            tot[5] += 1
    print(f"{label:10s} top1 {tot[1]}/{n} = {tot[1]/n*100:.2f}%   top5 {tot[5]}/{n} = {tot[5]/n*100:.2f}%")

print(f"--- matched comparison, same {len(word_rows)} word rows, beam k={K} ---")
acc_table(zeroshot_preds, "zero-shot")
acc_table(lora_preds, "lora")""")

code('''def fmt(cands):
    return "; ".join(f"{w}:{sc:.4f}" for w, sc in cands)

with open("/kaggle/working/qwen_matched_scores.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["context", "first_letter", "answer", "category", "zeroshot_preds", "lora_preds"])
    for r in word_rows:
        w.writerow([r["context"], r["first letter"], r["answer"], "word",
                    fmt(zeroshot_preds[id(r)]), fmt(lora_preds[id(r)])])
    for r in other_rows:
        w.writerow([r["context"], r["first letter"], r["answer"], categorize(r["answer"]), "", ""])

print("wrote /kaggle/working/qwen_matched_scores.csv")''')

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("qwen3b_matched_eval.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote qwen3b_matched_eval.ipynb,", len(cells), "cells")
