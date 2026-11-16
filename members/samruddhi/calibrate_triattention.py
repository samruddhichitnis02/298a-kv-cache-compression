"""Calibrate TriAttention frequency statistics for a model (faithful to the authors' scripts/calibrate.py).

TriAttention scores keys using per-head Q/K *frequency-domain* statistics computed
once per model. This reproduces WeianMao/triattention scripts/calibrate.py:
run one forward pass on plain text, hook every attention layer to capture the
post-RoPE query states, invert RoPE to recover the base (pre-RoPE) queries, treat
each head_dim vector as head_dim/2 complex frequency pairs, and store per head:

  * q_mean_complex  — mean of the complex queries over positions (the query "center")
  * q_abs_mean      — mean magnitude per frequency

The output .pt matches the authors' schema (metadata + per-head stats) so the same
scoring math can consume it. This replaces the online center-estimate proxy used
earlier, so measured results reflect TriAttention as designed.

    C:\\Python310\\python.exe members/samruddhi/calibrate_triattention.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output members/samruddhi/calibration/llama31_8b_stats.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# Natural-prose tasks (books, reports, articles, papers); code and synthetic tasks are left out.
PROSE_TASKS = {"narrativeqa", "gov_report", "multi_news", "qasper", "multifieldqa_en"}


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def invert_rope(rotated, cos, sin, scale):
    """Recover base (pre-RoPE) vectors. Mirrors pruning_utils.invert_rope (style='half')."""
    scale_t = torch.tensor(scale, device=rotated.device, dtype=rotated.dtype)
    base = rotated / scale_t
    cos_unit = cos / scale_t
    sin_unit = sin / scale_t
    return base * cos_unit - rotate_half(base) * sin_unit


def to_complex_pairs(tensor):
    """head_dim -> head_dim/2 complex pairs (front half = real, back half = imag)."""
    real_dtype = torch.float32 if tensor.dtype in (torch.bfloat16, torch.float16) else tensor.dtype
    t = tensor.to(dtype=real_dtype)
    fc = t.shape[-1] // 2
    return torch.complex(t[..., :fc].contiguous(), t[..., fc:].contiguous())


DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog. " * 4
    + "In information theory, entropy quantifies the uncertainty of a random variable. "
    + "Transformers process sequences in parallel using self-attention, where every token "
    + "attends to every earlier token through learned query, key and value projections. "
    + "The rotary position embedding rotates query and key vectors by an angle proportional "
    + "to their position, so that the dot product depends only on relative distance. "
)


def train_split_text(data_path, tok, max_length):
    """Natural calibration text, as the authors ask: whole LongBench documents from the train split.

    Train = every processed record that is in neither the validation (dev_ids.json) nor the test
    split (test_ids.json), and whose document does not appear in either of them. Documents come
    from the prose tasks, in record_id order, each used once, until max_length tokens are reached.
    """
    splits = REPO / "Data_pipeline" / "processed_longbench"
    dev = json.load(open(splits / "dev_ids.json"))
    dev = set(dev if isinstance(dev, list) else next(v for v in dev.values() if isinstance(v, list)))
    held_out = dev | set(json.load(open(splits / "test_ids.json"))["test_ids"])
    records = [json.loads(line) for line in open(data_path, encoding="utf-8") if line.strip()]
    held_docs = {r["context"] for r in records if r["record_id"] in held_out}
    seen, parts, n_tokens = set(), [], 0
    for r in sorted(records, key=lambda r: r["record_id"]):
        if (r["task"] not in PROSE_TASKS or r["record_id"] in held_out
                or r["context"] in held_docs or r["context"] in seen):
            continue
        seen.add(r["context"])
        parts.append(r["context"])
        n_tokens += len(tok(r["context"], add_special_tokens=False).input_ids)
        if n_tokens >= max_length:
            break
    print(f"calibration text: {len(parts)} train-split documents, {n_tokens} tokens", file=sys.stderr)
    return "\n\n".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--input", default=None,
                   help="plain-text calibration file (default: LongBench train-split documents, see train_split_text)")
    p.add_argument("--data", default=str(REPO / "data" / "processed_longbench" / "processed_longbench.jsonl"),
                   help="processed LongBench export used to build the default calibration text")
    p.add_argument("--repeated-sample", action="store_true",
                   help="old behaviour: repeat the built-in DEFAULT_TEXT. The authors advise against repetitive "
                        "calibration text; kept only to reproduce the Nov 10 stats file.")
    p.add_argument("--output", default=str(HERE / "calibration" / "llama31_8b_stats.pt"))
    p.add_argument("--max-length", type=int, default=32768,
                   help="calibration token length (authors' default 32768)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dtype = torch.bfloat16
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"loading {args.model} on {device}", file=sys.stderr)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                                 device_map=device, trust_remote_code=True)
    model.eval()

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)

    backbone = getattr(model, "model", model)
    attn_layers = [layer.self_attn for layer in backbone.layers]
    rotary = backbone.rotary_emb if hasattr(backbone, "rotary_emb") else attn_layers[0].rotary_emb
    attn_scale = float(getattr(rotary, "attention_scaling", 1.0))

    # --- calibration text ---
    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    elif args.repeated_sample:
        text = DEFAULT_TEXT
        while len(tok(text, add_special_tokens=False).input_ids) < args.max_length:
            text += DEFAULT_TEXT
    else:
        text = train_split_text(args.data, tok, args.max_length)
    input_ids = tok.encode(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
    seq_len = input_ids.shape[1]
    print(f"calibration tokens: {seq_len}", file=sys.stderr)

    # shared cos/sin for inversion, computed once
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    probe = torch.zeros(1, seq_len, head_dim, device=device, dtype=dtype)
    cos_table, sin_table = rotary(probe, position_ids)  # [1, seq_len, head_dim]
    cos_i = cos_table[0].unsqueeze(0)  # [1, seq_len, head_dim]
    sin_i = sin_table[0].unsqueeze(0)

    # Compute per-head complex stats INSIDE each hook and keep only the small result
    # (head_dim/2 complex numbers per head), so GPU memory stays flat across layers.
    stats, sampled = {}, []

    def make_hook(idx):
        def hook(module, a, kw):
            hs = a[0] if a else kw.get("hidden_states")
            if hs is None:
                return
            bsz, q_len, _ = hs.shape
            q = module.q_proj(hs).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            pos = torch.arange(q_len, device=hs.device).unsqueeze(0)
            pr = torch.zeros(1, q_len, head_dim, device=hs.device, dtype=hs.dtype)
            cos, sin = rotary(pr, pos)
            q_rot = ((q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))) * attn_scale
            # invert RoPE back to base queries, then reduce to per-head means immediately
            cos_h = cos_i[:, :q_len, :].unsqueeze(1)  # [1,1,q_len,head_dim]
            sin_h = sin_i[:, :q_len, :].unsqueeze(1)
            q_base = invert_rope(q_rot, cos_h, sin_h, attn_scale)
            for head_idx in range(num_heads):
                qc = to_complex_pairs(q_base[0, head_idx])  # [q_len, fc]
                qmean = qc.mean(dim=0)
                qabs = qc.abs().mean(dim=0)
                stats[f"layer{idx:02d}_head{head_idx:02d}"] = {
                    "q_mean_real": qmean.real.cpu(),
                    "q_mean_imag": qmean.imag.cpu(),
                    "q_abs_mean": qabs.cpu(),
                }
                sampled.append((idx, head_idx))
            del q, q_rot, q_base
        return hook

    handles = [attn.register_forward_pre_hook(make_hook(i), with_kwargs=True)
               for i, attn in enumerate(attn_layers)]
    print("forward pass...", file=sys.stderr)
    with torch.no_grad():
        backbone(input_ids)  # the hooks need attention inputs only; skipping lm_head avoids a seq_len x vocab logits tensor
    for h in handles:
        h.remove()

    payload = {
        "metadata": {
            "num_traces": 1, "head_dim": head_dim, "dtype": "bfloat16", "calibration_tokens": int(seq_len),
            "calibration_source": args.input or ("repeated DEFAULT_TEXT" if args.repeated_sample else "LongBench train split"),
            "rope_style": "half",
            "sampled_heads": [[int(l), int(h)] for l, h in sampled],
        },
        "stats": stats,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"saved {out} ({len(sampled)} heads)", file=sys.stderr)


if __name__ == "__main__":
    main()
