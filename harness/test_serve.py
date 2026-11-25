"""Checks for the packed cache and the serving driver. CPU only, no model download.

    python -m pytest -q harness/test_serve.py
"""
import sys
from pathlib import Path

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "members" / "anees"))

import kvpress  # noqa: E402,F401  (installs the masked-key attention patch)
from quantized_restorekv_press import cache_bytes, quantize_keys, quantize_values  # noqa: E402
from serve import (PackedLayer, cache_nbytes, clear_masks, eviction_ratio_for_budget, pack_cache,  # noqa: E402
                   pack_codes, unpack_codes)

HEADS, TOKENS, DIM = 4, 150, 64


def random_layer(dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(1, HEADS, TOKENS, DIM, generator=g).to(dtype)
    values = torch.randn(1, HEADS, TOKENS, DIM, generator=g).to(dtype)
    kept = torch.rand(HEADS, TOKENS, generator=g) < 0.4
    kept[0] = False
    kept[0, :3] = True          # a head that keeps almost nothing
    return keys, values, kept


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_code_packing_round_trips(bits):
    codes = torch.randint(0, 2**bits, (37, DIM), dtype=torch.uint8)
    packed = pack_codes(codes, bits)
    assert packed.shape == (37, DIM * bits // 8)
    assert torch.equal(unpack_codes(packed, bits), codes)


def test_lossless_pack_returns_the_kept_pairs():
    keys, values, kept = random_layer()
    layer = PackedLayer(keys, values, kept, bits=16)
    k, v = layer.unpack()
    for head in range(HEADS):
        n = int(kept[head].sum())
        assert torch.equal(k[0, head, :n], keys[0, head][kept[head]])
        assert torch.equal(v[0, head, :n], values[0, head][kept[head]])
        assert not k[0, head, n:].any()
    assert layer.nbytes() == int(kept.sum()) * 2 * DIM * 4       # float32 keys and values, nothing else


@pytest.mark.parametrize("bits", [2, 4])
def test_quantized_pack_matches_the_simulated_quantizer(bits):
    """Same grouping as quantized_restorekv_press, which rounds in place without packing."""
    keys, values, kept = random_layer()
    layer = PackedLayer(keys, values, kept, bits=bits)
    k, v = layer.unpack()
    want_k = quantize_keys(keys, kept.unsqueeze(0), bits)
    want_v = quantize_values(values, bits)
    for head in range(HEADS):
        n = int(kept[head].sum())
        assert torch.allclose(k[0, head, :n], want_k[0, head][kept[head]], atol=1e-5)
        assert torch.allclose(v[0, head, :n], want_v[0, head][kept[head]], atol=1e-5)


@pytest.mark.parametrize("bits", [2, 4])
def test_measured_bytes_match_the_computed_formula(bits):
    keys, values, kept = random_layer(torch.bfloat16)
    layer = PackedLayer(keys, values, kept, bits=bits)
    assert layer.nbytes() == cache_bytes([kept.unsqueeze(0)], DIM, bits)


@pytest.mark.parametrize("bits", [2, 4])
def test_quantization_error_is_bounded_by_the_step(bits):
    keys, values, kept = random_layer()
    layer = PackedLayer(keys, values, kept, bits=bits)
    k, _ = layer.unpack()
    for head in range(HEADS):
        n = int(kept[head].sum())
        original = keys[0, head][kept[head]]
        step = (original.amax() - original.amin()) / (2**bits - 1)
        assert (k[0, head, :n] - original).abs().max() <= step / 2 + 1e-4


def tiny_model():
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    return LlamaForCausalLM(config).eval()


def decode_logits(model, cache, position):
    ids = torch.tensor([[5, 9, 11]])
    positions = torch.arange(position, position + 3).unsqueeze(0)
    return model(input_ids=ids, past_key_values=cache, position_ids=positions).logits[0, -1]


def prefilled(model, tokens=96):
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=torch.randint(0, 128, (1, tokens), generator=torch.Generator().manual_seed(1)),
                    past_key_values=cache)
    return cache


def hide_pairs(model, tokens=96, share=0.6):
    g = torch.Generator().manual_seed(2)
    for layer in model.model.layers:
        hidden = torch.rand(2, tokens, generator=g) < share
        head, token = torch.nonzero(hidden, as_tuple=True)
        layer.self_attn.masked_key_indices = (torch.zeros_like(head), head, token)


def test_lossless_pack_decodes_like_the_masked_cache():
    """Hiding pairs with fake keys and really removing them must give the same next-token logits."""
    model = tiny_model()
    with torch.no_grad():
        cache = prefilled(model)
        hide_pairs(model)
        masked = decode_logits(model, cache, 96)

        cache = prefilled(model)
        hide_pairs(model)
        stats = pack_cache(model, cache, bits=16)
        packed = decode_logits(model, cache, 96)
    clear_masks(model)
    assert stats["kept_slots"] < stats["total_slots"]
    assert cache_nbytes(cache) < 0.6 * stats["total_slots"] * 2 * 32 * 4 + 3 * 2 * 3 * 32 * 4 * 2
    assert torch.allclose(masked, packed, atol=2e-3), (masked - packed).abs().max()
    assert masked.argmax() == packed.argmax()


def test_packed_cache_can_answer_again_after_crop():
    model = tiny_model()
    with torch.no_grad():
        cache = prefilled(model)
        hide_pairs(model)
        pack_cache(model, cache, bits=4)
        lengths = [cache.get_seq_length(i) for i in range(len(cache))]
        first = decode_logits(model, cache, 96)
        for layer, length in zip(cache.layers, lengths):
            layer.crop(length)
        second = decode_logits(model, cache, 96)
    clear_masks(model)
    assert len(set(lengths)) == 1                 # every layer reports the same length
    assert torch.equal(first, second)


@pytest.mark.parametrize("bits,budget", [(16, 0.1), (4, 0.1), (2, 0.1), (4, 0.05), (4, 0.5)])
def test_eviction_ratio_hits_the_byte_budget(bits, budget):
    model = tiny_model().to(torch.bfloat16)
    tokens, layers, heads, dim = 4000, 3, 2, 32
    ratio = eviction_ratio_for_budget(budget, bits, tokens, model)
    g = torch.Generator().manual_seed(3)
    kept = [torch.rand(1, heads, tokens, generator=g) >= ratio for _ in range(layers)]
    full = layers * heads * tokens * 2 * dim * 2
    if bits == 16:
        used = sum(int(k.sum()) for k in kept) * 2 * dim * 2
    else:
        used = cache_bytes(kept, dim, bits)
    if ratio == 0:
        assert used <= budget * full
    else:
        assert abs(used / full - budget) < 0.01
