"""Swappable causal-LM architecture registry -- trains from scratch (random init), never
pulls pretrained weights. We only import the *config*/*model class* from each family and
build our own small config with our own vocab_size; transformers never hits the network for
this (from_config, not from_pretrained).

Add a new family: drop a name -> build_fn entry in MODEL_REGISTRY. build_fn takes the
normalized hyperparams below and returns a randomly-initialized nn.Module.

Usage:
    from models import build_model
    model = build_model("gpt2", vocab_size=len(vocab), n_layer=6, n_head=8, n_embd=512, seq_len=256)
    model = build_model("qwen2", vocab_size=len(vocab), n_layer=6, n_head=8, n_embd=512, seq_len=256)
"""


def _build_gpt2(vocab_size, n_layer, n_head, n_embd, seq_len):
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_len,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )
    return GPT2LMHeadModel(cfg)  # random init -- no .from_pretrained


def _build_qwen2(vocab_size, n_layer, n_head, n_embd, seq_len):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(
        vocab_size=vocab_size,
        max_position_embeddings=seq_len,
        hidden_size=n_embd,
        intermediate_size=n_embd * 4,
        num_hidden_layers=n_layer,
        num_attention_heads=n_head,
        num_key_value_heads=n_head,  # set < n_head for GQA if you want fewer KV heads
    )
    return Qwen2ForCausalLM(cfg)  # random init -- no .from_pretrained


MODEL_REGISTRY = {
    "gpt2": _build_gpt2,
    "qwen2": _build_qwen2,
}


def build_model(name, vocab_size, n_layer=6, n_head=8, n_embd=512, seq_len=256):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}, pick from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](vocab_size, n_layer, n_head, n_embd, seq_len)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
