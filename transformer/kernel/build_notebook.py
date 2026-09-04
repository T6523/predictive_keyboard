"""Generates train.ipynb from the checked-in .py source (models.py, vocab.py, data.py) plus
the cell code below. Source of truth is this file, not the .ipynb -- re-run after editing
models.py / vocab.py / data.py or the cell strings:

    python3 transformer/kernel/build_notebook.py
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).resolve().parent / "train.ipynb"


def embed(fname):
    src = (ROOT / fname).read_text()
    return f"%%writefile {fname}\n{src}"


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Predictive-keyboard transformer -- from scratch, resumable, model-swappable

Trains a causal LM (random init, no pretrained weights) on the Gigaword word-level corpus,
with the softmax masked to the target word's first-letter bucket at every training position
(same contract as `scripts/eval_ngram.py`).

**Switch architecture**: change `MODEL_NAME` below (`models.MODEL_REGISTRY`). Different
architecture = different checkpoint file (`checkpoints/<MODEL_NAME>.pt`), fresh random init --
there's no weight transfer between architectures, only same-architecture resume.

**Resume across sessions**: attach a previous run's checkpoint as a Kaggle dataset input (any
slug, this notebook globs `/kaggle/input/*/<MODEL_NAME>.pt`). Vocab is loaded from the
checkpoint on resume (never rebuilt -- token ids must stay stable). Config in the checkpoint
must match this run's CONFIG cell exactly, or it refuses to resume (silently resuming into
mismatched hyperparams is the classic footgun).

**Autosave**: checkpoint is written to `/kaggle/working/checkpoints/` every `SAVE_EVERY_STEPS`,
not just at the end -- so a killed/timed-out session still leaves a usable checkpoint in the
kernel's committed Output. `TIME_BUDGET_SEC` cuts training loose *before* Kaggle's own session
limit so the script always exits cleanly (required for Kaggle to commit Output at all).

**Double descent**: `SUBSET_FRAC` shrinks the training set so a small model can reach the
interpolation threshold (train loss -> ~0) in reasonable wall-clock time -- the epoch-wise
double-descent regime (Nakkiran et al. 2019). Train past that point (raise `EPOCHS` / resume
across sessions) and watch the heldout-loss curve plotted at the end: dip, bump, redip.

**This run**: full corpus (`SUBSET_FRAC=1.0`), 1 epoch, ~497M-param model (`n_embd=1152,
n_layer=24, n_head=18` -- GPT2-medium widened for this task's ~99k-word vocab). Baseline
(random-init) accuracy is measured right after the model is built, before any training, so
it's directly comparable to the post-training number logged after the final checkpoint save.""")

code("""import sys, os, glob, time, random
sys.path.insert(0, ".")
""")

code(embed("models.py"))
code(embed("vocab.py"))
code(embed("data.py"))

code("""import torch
import torch.nn.functional as F

from models import build_model, n_params
from vocab import build_vocab, build_bucket_table, BOS, EOS, BUCKETS
from data import tokenize_corpus, n_blocks, block

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    # ponytail: Kaggle's GPU allocation is random and can hand out old hardware (P100, sm_60)
    # the preinstalled torch build doesn't support at all -- fail soft to CPU instead of crashing
    # on the first forward pass. Re-push to get reallocated if you land here; no code fix for it.
    cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
    if cap not in torch.cuda.get_arch_list():
        print(f"GPU {torch.cuda.get_device_name(0)} ({cap}) unsupported by installed torch "
              f"(supports {torch.cuda.get_arch_list()}) -- falling back to CPU. Re-push for a "
              f"different GPU allocation.")
        device = "cpu"
print("device:", device, "| gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
""")

md("""## Config -- single source of truth for this run

Real run: full corpus, 1 epoch, ~497M params. `MAX_STEPS` is computed after the dataset is
built (one epoch's worth of steps at `BATCH_SIZE`), not hardcoded -- see the Dataset cell.""")

code("""MODEL_NAME = "gpt2"          # switch: "gpt2" | "qwen2" (models.MODEL_REGISTRY)

print("/kaggle/input contents:", glob.glob("/kaggle/input/*"))
print("/kaggle/input/*/* :", glob.glob("/kaggle/input/*/*"))
_data_files = glob.glob("/kaggle/input/*/train.src.tok") + glob.glob("/kaggle/input/**/train.src.tok", recursive=True)
assert _data_files, "attach the gigaword dataset as a Kaggle input (Add Data)"
DATA_DIR = os.path.dirname(_data_files[0])
TRAIN_PATH = f"{DATA_DIR}/train.src.tok"
EVAL_PATH = f"{DATA_DIR}/devv_eval.csv"

# any attached input dataset containing checkpoints/<MODEL_NAME>.pt resumes from it
_resume = glob.glob(f"/kaggle/input/*/checkpoints/{MODEL_NAME}.pt") + glob.glob(f"/kaggle/input/*/{MODEL_NAME}.pt")
CKPT_IN = _resume[0] if _resume else None
CKPT_OUT_DIR = "/kaggle/working/checkpoints"
CKPT_OUT = f"{CKPT_OUT_DIR}/{MODEL_NAME}.pt"

SEQ_LEN, N_LAYER, N_HEAD, N_EMBD = 256, 24, 18, 1152   # ~497M params (99k vocab, tied embed)
MIN_COUNT = 1                  # keep full vocab: cutting costs 1-5 accuracy pts for ~no speedup
                                # (bucket-masked loss already caps softmax to ~1/36 of vocab/step)

SUBSET_FRAC = 1.0              # full corpus
EPOCHS = 1                     # MAX_STEPS computed from this once train_ds is built (Dataset cell)
BATCH_SIZE = 16                # fits ~497M params + seq_len 256 on a 16GB GPU w/ grad checkpointing
LR = 1.5e-4                    # scaled down from the test run's 3e-4 for this model size
TIME_BUDGET_SEC = 8 * 3600     # per-session budget, under Kaggle's ~9-12h cap -- resumes across
                                # sessions via the attached checkpoint dataset until MAX_STEPS
SAVE_EVERY_STEPS = 500
EVAL_EVERY_STEPS = 500
SEED = 0

print(f"model={MODEL_NAME} resume={'yes: ' + CKPT_IN if CKPT_IN else 'no (fresh init)'}")
""")

md("## Vocab -- rebuilt only for a fresh run; loaded from checkpoint on resume")

code("""if CKPT_IN:
    ckpt = torch.load(CKPT_IN, map_location="cpu")
    vocab, id_to_tok = ckpt["vocab"], ckpt["id_to_tok"]
    print(f"resumed vocab: {len(vocab)} tokens (loaded from checkpoint, not rebuilt)")
else:
    ckpt = None
    vocab, id_to_tok = build_vocab(TRAIN_PATH, min_count=MIN_COUNT)
    print(f"fresh vocab: {len(vocab)} tokens")

bos_id, eos_id = vocab[BOS], vocab[EOS]

bucket_table = build_bucket_table(id_to_tok)               # id -> 0..35, or -1
bucket_table_t = torch.tensor(bucket_table, device=device)

# bucket_cols[b] = vocab ids in bucket b; bucket_pos[id] = that id's row within its bucket --
# together these let the loss gather a *small* (n, |bucket|) slice instead of masking the full
# (B, T, V) logits (that's a 2GB+ bool tensor at this vocab size -- OOM, not just slow)
bucket_cols = [torch.tensor([i for i, b in enumerate(bucket_table) if b == k], device=device, dtype=torch.long)
               for k in range(len(BUCKETS))]
bucket_pos = torch.full((len(vocab),), -1, dtype=torch.long)
for k, cols in enumerate(bucket_cols):
    bucket_pos[cols.cpu()] = torch.arange(len(cols))
bucket_pos = bucket_pos.to(device)
""")

md("""## Dataset

Causal-LM block packing over `<s>`/`</s>`-delimited lines. A fixed ~2% of lines (by index,
`data.HELDOUT_EVERY`) is held out from training for eval-loss tracking -- needed to actually
see a double-descent curve, not just a final number. `SUBSET_FRAC` subsamples within each
split so the test run and a full run share the same code path.""")

code("""t0 = time.time()
train_ids = tokenize_corpus(TRAIN_PATH, vocab, frac=SUBSET_FRAC, seed=SEED, split="train")
heldout_ids = tokenize_corpus(TRAIN_PATH, vocab, frac=SUBSET_FRAC, seed=SEED, split="heldout")
train_ids = torch.tensor(train_ids, dtype=torch.long)
heldout_ids = torch.tensor(heldout_ids, dtype=torch.long)
print(f"train: {len(train_ids)} tokens -> {n_blocks(len(train_ids), SEQ_LEN)} blocks | "
      f"heldout: {len(heldout_ids)} tokens -> {n_blocks(len(heldout_ids), SEQ_LEN)} blocks "
      f"({time.time()-t0:.1f}s)")

class BlockDataset(torch.utils.data.Dataset):
    def __init__(self, ids, seq_len):
        self.ids, self.seq_len = ids, seq_len
        self.n = n_blocks(len(ids), seq_len)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.seq_len
        chunk = self.ids[s : s + self.seq_len + 1]
        return chunk[:-1], chunk[1:]

train_ds = BlockDataset(train_ids, SEQ_LEN)
heldout_ds = BlockDataset(heldout_ids, SEQ_LEN)
assert len(train_ds) > 0 and len(heldout_ds) > 0, "SUBSET_FRAC too small for this SEQ_LEN -- raise one of them"

MAX_STEPS = EPOCHS * (len(train_ds) // BATCH_SIZE)  # absolute target -- stays fixed across resumed sessions
print(f"{EPOCHS} epoch(s) -> {MAX_STEPS} steps total @ batch {BATCH_SIZE}")
""")

md("## Model -- fresh init, or resume (state dict + optimizer + step count)")

code("""model = build_model(MODEL_NAME, vocab_size=len(vocab), n_layer=N_LAYER, n_head=N_HEAD,
                    n_embd=N_EMBD, seq_len=SEQ_LEN).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
start_step = 0

cur_config = dict(model_name=MODEL_NAME, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
                   seq_len=SEQ_LEN, vocab_size=len(vocab))

if ckpt is not None:
    if ckpt["config"] != cur_config:
        raise ValueError(
            f"checkpoint config {ckpt['config']} != this run's config {cur_config} -- "
            f"can't resume into mismatched hyperparams. Match CONFIG to the checkpoint, "
            f"or clear CKPT_IN to start {MODEL_NAME} fresh."
        )
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    if device == "cuda" and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    start_step = ckpt["step"]
    print(f"resumed {MODEL_NAME} @ step {start_step}")
else:
    print(f"fresh {MODEL_NAME}: {n_params(model)/1e6:.1f}M params, random init")

model.gradient_checkpointing_enable()
""")

md("""## Eval helpers + baseline

Mirrors `scripts/eval_ngram.py`'s `predict(context_tokens, letter)` contract: last position's
logits, masked to the letter's bucket, argmax. Symbol/`[UNK]` first letters are routed
elsewhere (`scripts/symbol_predict.py`) -- this only ever sees alnum letters.

Defined *before* training so the baseline (random-init) accuracy below is directly comparable
to the post-training number logged after the final checkpoint save.""")

code("""# eval-only: full (bucket, vocab) bool mask -- fine at this size (~3.5MB) as a one-off,
# unlike the per-position (B, T, V) version that OOMs during training
bucket_bool_mask = torch.zeros(len(BUCKETS), len(vocab), dtype=torch.bool, device=device)
for b, cols in enumerate(bucket_cols):
    bucket_bool_mask[b, cols] = True

@torch.no_grad()
def predict(context_tokens, letter, max_ctx=SEQ_LEN - 1):
    model.eval()
    if letter.lower() not in BUCKETS:
        return None
    ids = [bos_id] + [vocab.get(t, bos_id) for t in context_tokens][-max_ctx:]
    x = torch.tensor([ids], device=device)
    logits = model(x).logits[0, -1]
    b = BUCKETS.index(letter.lower())
    logits = logits.masked_fill(~bucket_bool_mask[b], float("-inf"))
    best = logits.argmax().item()
    return id_to_tok[best] if logits[best] != float("-inf") else None

import csv

def run_eval(path, limit=None):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    rows = [r for r in rows if r["first letter"].isalnum()]
    correct = sum(predict(r["context"].split(), r["first letter"]) == r["answer"] for r in rows)
    acc = correct / max(len(rows), 1)
    print(f"accuracy: {correct}/{len(rows)} = {acc:.4f}")
    return acc

EVAL_LIMIT = 3000  # sample, not the full 67k-row set -- predict() is one forward pass per row,
                    # unbatched; full-set eval at this model size is a real time cost, not free
""")

code("""print("--- baseline (random init, before any training) ---")
baseline_acc = run_eval(EVAL_PATH, limit=EVAL_LIMIT) if ckpt is None else None
if baseline_acc is None:
    print("(resumed run -- baseline was measured in the session that created this checkpoint)")
""")

md("""## Masked loss

Restricts the softmax to the target's first-letter bucket at every position -- grouped by
bucket (36 small gathers) rather than one `(B, T, V)` boolean mask, which OOMs at this vocab
size.""")

code("""def masked_ce_loss(logits, targets, reduction="mean"):
    B, T, V = logits.shape
    logits = logits.reshape(-1, V)
    targets = targets.reshape(-1)
    buckets = bucket_table_t[targets]
    valid = buckets >= 0
    total, n = logits.new_zeros(()), 0
    for b in range(len(BUCKETS)):
        sel = ((buckets == b) & valid).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        cols = bucket_cols[b]
        sub_logits = logits[sel][:, cols]
        sub_targets = bucket_pos[targets[sel]]
        total = total + F.cross_entropy(sub_logits, sub_targets, reduction="sum")
        n += sel.numel()
    if reduction == "sum":
        return total, n
    return total / max(n, 1)


@torch.no_grad()
def eval_loss(n_batches=10):
    model.eval()
    loader = torch.utils.data.DataLoader(heldout_ds, batch_size=BATCH_SIZE, shuffle=True)
    tot, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(x).logits
            s, cnt = masked_ce_loss(logits, y, reduction="sum")
        tot += s.item(); n += cnt
    model.train()
    return tot / max(n, 1)
""")

md("""## Train

Hard-stops at `MAX_STEPS` or `TIME_BUDGET_SEC`, whichever first. Saves every
`SAVE_EVERY_STEPS` so a killed session still leaves a usable checkpoint. Logs train + heldout
loss every `EVAL_EVERY_STEPS` -- this history is the double-descent curve, plotted below.""")

code("""def save_ckpt(step):
    os.makedirs(CKPT_OUT_DIR, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict() if device == "cuda" else None,
        "vocab": vocab, "id_to_tok": id_to_tok,
        "step": step,
        "config": cur_config,
        "history": history,
    }, CKPT_OUT)
    print(f"  saved -> {CKPT_OUT} @ step {step}")

history = ckpt["history"] if ckpt is not None and "history" in ckpt else []  # [(step, train_loss, heldout_loss)]

def make_loader(step):
    g = torch.Generator().manual_seed(SEED + step)  # ponytail: reshuffled per remake, not a
    return torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                        drop_last=True, generator=g)
    # byte-exact resumed iterator -- fine for LM training over many blocks; upgrade to
    # persisting sampler state if exact resume-mid-epoch ever matters

model.train()
t0 = time.time()
step = start_step
loader = make_loader(step)
it = iter(loader)
while step < MAX_STEPS:
    try:
        x, y = next(it)
    except StopIteration:
        it = iter(make_loader(step))
        x, y = next(it)
    x, y = x.to(device), y.to(device)
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
        logits = model(x).logits
        loss = masked_ce_loss(logits, y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    step += 1

    if step % EVAL_EVERY_STEPS == 0 or step == MAX_STEPS:
        hl = eval_loss()
        history.append((step, loss.item(), hl))
        print(f"step {step}/{MAX_STEPS} train_loss {loss.item():.3f} heldout_loss {hl:.3f} "
              f"({time.time()-t0:.0f}s)")
    if step % SAVE_EVERY_STEPS == 0:
        save_ckpt(step)
    if time.time() - t0 > TIME_BUDGET_SEC:
        print(f"time budget ({TIME_BUDGET_SEC}s) hit @ step {step}, stopping")
        break

save_ckpt(step)
print("training done.")

print("--- eval after training (same sample as baseline) ---")
trained_acc = run_eval(EVAL_PATH, limit=EVAL_LIMIT)
if baseline_acc is not None:
    print(f"baseline -> trained: {baseline_acc:.4f} -> {trained_acc:.4f} "
          f"({'+' if trained_acc >= baseline_acc else ''}{trained_acc - baseline_acc:.4f})")
""")

code("""import matplotlib.pyplot as plt
if history:
    steps, tr, ho = zip(*history)
    plt.plot(steps, tr, label="train loss")
    plt.plot(steps, ho, label="heldout loss")
    plt.xlabel("step"); plt.ylabel("masked CE loss"); plt.legend()
    plt.title(f"{MODEL_NAME} -- watch for the double-descent bump as training extends past interpolation")
    plt.show()
else:
    print("no history yet (single short run) -- accumulates across resumed sessions")
""")

nb["cells"] = cells
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
