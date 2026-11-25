"""Run:  python -m pytest -q members/manjot/test_rdkv.py      (CPU, ~30 s, no model download)"""
import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rdkv_press import RDKVPress, allocate_bits, fake_quant  # noqa: E402

BITS = torch.tensor([0, 2, 4, 8, 16.0])


def test_fake_quant_error_shrinks_with_bits():
    x = torch.randn(4, 64, 128)
    errs = [(x - fake_quant(x, b, dim=-1)).pow(2).mean().item() for b in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2] > 0
    assert torch.equal(fake_quant(x, 16, dim=-1), x)


@pytest.mark.parametrize("f", [0.25, 0.2, 0.1, 0.05])
def test_allocator_hits_budget(f):
    torch.manual_seed(0)
    N, U = 4, 3000
    w = torch.rand(N, U).pow(4) + 1e-4
    sig = torch.rand(N, U) + 0.5
    eps = torch.stack([sig, sig / 4, sig / 16, sig / 256, sig * 0], -1)
    target = torch.full((N,), 16 * f)
    idx = allocate_bits(w[..., None] * eps, BITS, target)
    avg = BITS[idx].mean(-1)
    assert (avg <= target + 1e-6).all()
    assert ((target - avg) / target < 0.02).all()
    # mixed allocation: some evicted, some quantised
    assert (idx == 0).any() and ((idx > 0) & (idx < 4)).any()


@pytest.fixture(scope="module")
def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=4096,
                      attn_implementation="sdpa")
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


@pytest.mark.parametrize("ratio", [0.75, 0.8, 0.9, 0.95])
def test_end_to_end_tiny_llama(tiny_llama, ratio):
    ids = torch.randint(0, 256, (1, 1024))
    press = RDKVPress(compression_ratio=ratio)
    with torch.no_grad(), press(tiny_llama):
        out = tiny_llama(ids, use_cache=True)
    s = press.summary()
    assert s["kv_cache_gb_payload"] > 0
    payload_frac = s["kv_cache_gb_payload"] / s["kv_cache_gb_full"]
    assert payload_frac <= (1 - ratio) + 1e-3, payload_frac
    assert payload_frac >= (1 - ratio) * 0.9, payload_frac  # budget actually used
    assert torch.isfinite(out.logits).all()
    # a decode step after compression must run (exercises kvpress masked-key patch)
    with torch.no_grad():
        nxt = tiny_llama(ids[:, -1:], past_key_values=out.past_key_values, use_cache=True)
    assert torch.isfinite(nxt.logits).all()
    print(ratio, s)


# ---- packed storage (rdkv_pack.py) --------------------------------------------------------------
from rdkv_pack import nbytes, pack_layer, unpack_layer  # noqa: E402


def _mixed_layer(dtype=torch.float32):
    g = torch.Generator().manual_seed(0)
    H, T, d = 3, 90, 64
    keys = torch.randn(1, H, T, d, generator=g).to(dtype)
    values = torch.randn(1, H, T, d, generator=g).to(dtype)
    widths = torch.tensor([0, 2, 4, 8, 16])
    value_bits = widths[torch.randint(0, 5, (H, T), generator=g)]
    value_bits[:, -8:] = 16                                    # recent window stays full precision
    key_bits = widths[torch.randint(0, 5, (H, d), generator=g)]
    return keys, values, value_bits, key_bits, value_bits > 0


def test_packed_layer_round_trips_within_one_quantisation_step():
    keys, values, value_bits, key_bits, kept = _mixed_layer()
    packed = pack_layer(keys, values, value_bits, key_bits, kept)
    k, v = unpack_layer(packed, keys, values, kept)
    for h in range(keys.shape[1]):
        for bits in (2, 4, 8):
            rows = kept[h] & (value_bits[h] == bits)
            original, got = values[0, h][rows], v[0, h][rows]
            step = (original.amax(1) - original.amin(1)) / (2**bits - 1)
            assert ((got - original).abs().amax(1) <= step / 2 + 1e-4).all()
            channels = key_bits[h] == bits
            original, got = keys[0, h][kept[h]][:, channels], k[0, h][kept[h]][:, channels]
            step = (original.amax(0) - original.amin(0)) / (2**bits - 1)
            assert ((got - original).abs().amax(0) <= step / 2 + 1e-4).all()
        full = kept[h] & (value_bits[h] == 16)
        assert torch.equal(v[0, h][full], values[0, h][full])                       # 16-bit values untouched
        assert not k[0, h][kept[h]][:, key_bits[h] == 0].any()                      # dropped channels are zero
        assert torch.equal(k[0, h][kept[h]][:, key_bits[h] == 16], keys[0, h][kept[h]][:, key_bits[h] == 16])
        assert torch.equal(v[0, h][~kept[h]], values[0, h][~kept[h]])               # evicted tokens left alone


def test_packed_bytes_follow_the_bit_allocation():
    keys, values, value_bits, key_bits, kept = _mixed_layer(torch.bfloat16)
    packed = pack_layer(keys, values, value_bits, key_bits, kept)
    d = keys.shape[-1]
    expected = 0
    for h in range(keys.shape[1]):
        n = int(kept[h].sum())
        for bits in (2, 4, 8):
            tokens = int((kept[h] & (value_bits[h] == bits)).sum())
            channels = int((key_bits[h] == bits).sum())
            expected += tokens * (d * bits // 8 + 4)                                 # codes + scale and minimum
            expected += channels * (-(-n * bits // 8) + 4)
        expected += int((kept[h] & (value_bits[h] == 16)).sum()) * d * 2
        expected += int((key_bits[h] == 16).sum()) * n * 2
        expected += n + d                                                            # one tag byte each
    assert nbytes(packed) == expected


@pytest.mark.parametrize("ratio", [0.75, 0.9])
def test_press_measures_packed_size_close_to_its_computed_size(ratio):
    from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
    model = LlamaForCausalLM(config).eval()
    ids = torch.randint(0, 128, (1, 600))
    press = RDKVPress(compression_ratio=ratio, measure_packed=True)
    with torch.no_grad(), press(model):
        model.model(input_ids=ids, past_key_values=DynamicCache())
    summary = press.summary()
    assert summary["packed_gb"] > 0
    # fp32 test model: payload is counted against 16-bit storage, so compare shapes, not exact bytes
    assert 0.5 < summary["packed_gb"] / (2 * summary["kv_cache_gb"]) < 1.3
