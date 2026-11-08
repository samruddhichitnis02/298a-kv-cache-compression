"""TurboQuant as a kvpress press, so it runs in the shared harness (harness/run_longbench.py).

After each layer's forward pass over the document, the document's keys and values are
packed with Vedant's `TurboQuantCacheLayer` (src/turboquant/kv_cache.py) and replaced by
their decoded values:
- keys: TurboQuant-Prod ((bits-1)-bit MSE codes plus a 1-bit QJL residual), or
  TurboQuant-MSE at `bits` when `qjl_keys=False`;
- values: TurboQuant-MSE at `bits`;
- vector norms (and QJL residual norms) in fp16.

This follows the shared protocol used for every other method:
- the document's own attention during prefill runs at full precision, and only what is
  stored is quantized (as in the TurboQuant and KIVI papers);
- the question and the generated answer stay in bf16, uncompressed;
- the cache holds the decoded bf16 tensors, so memory is computed, not measured, as for
  KVzip and RestoreKV. `context_bytes_per_token` gives the packed size, which the harness
  uses for `kv_cache_gb`; `stats["packed_bytes"]` is the exact count from the last record.

Seeds per layer follow `TransformersTurboQuantCache`, so both paths quantize identically.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from kvpress.presses.base_press import BasePress

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from turboquant.kv_cache import TurboQuantCacheLayer  # noqa: E402

NORM_BYTES = 2  # fp16


def head_token_bytes(head_dim: int, bits: int, qjl_keys: bool = True) -> float:
    """Packed bytes for one token's key and value in one KV head."""
    value_bytes = head_dim * bits / 8 + NORM_BYTES
    # QJL keys: (bits-1)-bit MSE codes + 1 sign bit per coordinate, plus vector and residual norms
    key_bytes = head_dim * bits / 8 + (2 if qjl_keys else 1) * NORM_BYTES
    return key_bytes + value_bytes


@dataclass
class TurboQuantPress(BasePress):
    bits: int = 3
    qjl_keys: bool = True
    seed: int = 42
    stats: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.bits not in (2, 3, 4):
            raise ValueError("bits must be 2, 3 or 4")
        if self.qjl_keys and self.bits < 2:
            raise ValueError("QJL keys need at least 2 bits")

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        layer = TurboQuantCacheLayer(
            keys.shape[-1], self.bits, qjl_keys=self.qjl_keys,
            seed=self.seed + module.layer_idx * 100_003,
        )
        layer.update(keys, values)
        if module.layer_idx == 0:
            self.stats = {"bits": self.bits, "qjl_keys": self.qjl_keys, "packed_bytes": 0}
        self.stats["packed_bytes"] += layer.storage_stats().compressed_bytes
        return layer._decoded_keys, layer._decoded_values

    def context_bytes_per_token(self, config) -> float:
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        per_head = head_token_bytes(head_dim, self.bits, self.qjl_keys)
        return config.num_hidden_layers * config.num_key_value_heads * per_head
