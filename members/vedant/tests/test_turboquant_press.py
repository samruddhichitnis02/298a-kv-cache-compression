"""The kvpress TurboQuant press and the cache changes for the shared harness."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from turboquant.kv_cache import TurboQuantCacheLayer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("kvpress")
from turboquant_press import TurboQuantPress, head_token_bytes  # noqa: E402

LLAMA = SimpleNamespace(num_hidden_layers=32, num_key_value_heads=8, head_dim=128)


def test_bytes_per_token_match_the_forecast_table() -> None:
    # STATUS.md: 17,920 B/token for QJL keys at 2 bits, 17,408 for MSE keys at 2 bits
    assert TurboQuantPress(bits=2).context_bytes_per_token(LLAMA) == 17_920
    assert TurboQuantPress(bits=2, qjl_keys=False).context_bytes_per_token(LLAMA) == 17_408
    assert TurboQuantPress(bits=4).context_bytes_per_token(LLAMA) == 34_304


@pytest.mark.parametrize("bits,qjl", [(2, True), (3, True), (4, False)])
def test_formula_matches_packed_bytes(bits: int, qjl: bool) -> None:
    keys = torch.randn(1, 2, 64, 128)
    layer = TurboQuantCacheLayer(128, bits, qjl_keys=qjl)
    layer.update(keys, keys.clone())
    per_head_token = layer.storage_stats().compressed_bytes / (2 * 64)
    assert per_head_token == head_token_bytes(128, bits, qjl)


def test_prefill_attends_at_full_precision_and_stores_quantized() -> None:
    x = torch.randn(1, 2, 8, 16)
    layer = TurboQuantCacheLayer(16, 3, seed=1)
    keys, values = layer.update(x, x)
    assert torch.equal(keys, x) and torch.equal(values, x)
    assert not torch.allclose(layer._decoded_keys, x)
    step = torch.randn(1, 2, 1, 16)
    keys, _ = layer.update(step, step)
    assert keys.shape == (1, 2, 9, 16)
    assert torch.equal(keys[:, :, :8], layer._decoded_keys[:, :, :8])
    assert torch.equal(keys[:, :, 8:], step)


def test_kept_decoded_copy_equals_full_dequantize() -> None:
    layer = TurboQuantCacheLayer(16, 3, seed=2)
    for length in (6, 1, 1, 3):
        x = torch.randn(1, 2, length, 16)
        layer.update(x, x * 0.5)
    keys, values = layer.dequantize()
    assert torch.equal(keys, layer._decoded_keys)
    assert torch.equal(values, layer._decoded_values)


def test_press_quantizes_like_the_cache_and_counts_bytes() -> None:
    press = TurboQuantPress(bits=3, seed=42)
    keys = torch.randn(1, 2, 10, 16, dtype=torch.bfloat16)
    values = torch.randn(1, 2, 10, 16, dtype=torch.bfloat16)
    module = SimpleNamespace(layer_idx=0)
    out_keys, out_values = press.compress(module, None, keys, values, None, {})
    assert out_keys.dtype == torch.bfloat16 and out_keys.shape == keys.shape
    reference = TurboQuantCacheLayer(16, 3, seed=42)
    reference.update(keys, values)
    assert torch.equal(out_keys, reference._decoded_keys)
    assert torch.equal(out_values, reference._decoded_values)
    assert press.stats["packed_bytes"] == 2 * 10 * head_token_bytes(16, 3)
    error = (out_keys.float() - keys.float()).norm() / keys.float().norm()
    assert 0 < error < 0.6
