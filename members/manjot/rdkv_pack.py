"""Real packed storage for an RDKV-compressed layer, so its size is measured instead of computed.

RDKV gives every value token and every key channel its own bit-width from {0, 2, 4, 8, 16}. The press
normally rounds tensors in place and hides evicted tokens, so nothing gets smaller. `pack_layer` writes
one layer into the buffers a real system would hold:

    values  kept tokens grouped by bit-width; 2/4/8-bit rows as packed codes with one scale and one
            minimum per token, 16-bit rows as they are
    keys    channels grouped by bit-width, over the kept tokens only; 2/4/8-bit channels as packed codes
            with one scale and one minimum per channel, 16-bit channels as they are, 0-bit channels dropped
    tags    one byte per kept token and per channel saying which bit-width it has

Scales and minimums are stored in the model's dtype. `nbytes(packed)` is the size of those buffers, and
`unpack_layer` rebuilds the tensors the model attends to. When the press runs with `measure_packed=True`
it uses the unpacked tensors, so the scores come from exactly what is stored.

Evicted tokens need no position record: keys in the cache already carry their rotary position.
"""

import torch


def _pack_codes(codes, bits):
    """uint8 codes below 2**bits, (rows, n) -> (rows, ceil(n * bits / 8)) bytes."""
    per_byte = 8 // bits
    if per_byte == 1:
        return codes
    rows, n = codes.shape
    pad = (-n) % per_byte
    if pad:
        codes = torch.cat([codes, codes.new_zeros(rows, pad)], dim=1)
    shifts = torch.arange(per_byte, device=codes.device, dtype=torch.uint8) * bits
    return (codes.view(rows, -1, per_byte) << shifts).sum(dim=-1, dtype=torch.uint8)


def _unpack_codes(packed, bits, n):
    per_byte = 8 // bits
    if per_byte == 1:
        return packed
    shifts = torch.arange(per_byte, device=packed.device, dtype=torch.uint8) * bits
    return ((packed.unsqueeze(-1) >> shifts) & (2**bits - 1)).reshape(packed.shape[0], -1)[:, :n]


def _quantize_rows(x, bits, dtype):
    """Min-max quantize each row of x (rows, n) to `bits`; scale and minimum stored in `dtype`."""
    levels = 2**bits - 1
    low = x.amin(dim=1, keepdim=True)
    scale = (x.amax(dim=1, keepdim=True) - low).clamp_min(1e-8) / levels
    low, scale = low.to(dtype), scale.to(dtype)
    codes = ((x - low.float()) / scale.float().clamp_min(1e-12)).round().clamp_(0, levels).to(torch.uint8)
    return {"codes": _pack_codes(codes, bits), "low": low.squeeze(1), "scale": scale.squeeze(1)}


def _dequantize_rows(store, bits, n, dtype):
    codes = _unpack_codes(store["codes"], bits, n).float()
    return (codes * store["scale"].float().unsqueeze(1) + store["low"].float().unsqueeze(1)).to(dtype)


@torch.no_grad()
def pack_layer(keys, values, value_bits, key_bits, kept):
    """Pack one layer.

    keys, values  (1, heads, tokens, dim) in the model's dtype
    value_bits    (heads, tokens) int: bit-width of each token's value; 0 where the token is evicted
    key_bits      (heads, dim) int: bit-width of each key channel; 0 where the channel is dropped
    kept          (heads, tokens) bool: tokens that stay in the cache
    Returns a list with one dict of buffers per head.
    """
    dtype = keys.dtype
    heads = []
    for h in range(keys.shape[1]):
        keep = kept[h]
        v = values[0, h][keep]                                 # (kept, dim)
        k = keys[0, h][keep]                                   # (kept, dim)
        v_tag = value_bits[h][keep].to(torch.uint8)            # bit-width of each kept token's value
        k_tag = key_bits[h].to(torch.uint8)                    # bit-width of each channel
        store = {"n": int(keep.sum()), "v_tag": v_tag, "k_tag": k_tag, "v": {}, "k": {}}
        for bits in (2, 4, 8):
            rows = v_tag == bits
            if bool(rows.any()):
                store["v"][bits] = _quantize_rows(v[rows].float(), bits, dtype)
            channels = k_tag == bits
            if bool(channels.any()) and store["n"]:
                store["k"][bits] = _quantize_rows(k[:, channels].float().t().contiguous(), bits, dtype)
        store["v"][16] = v[v_tag == 16].clone()
        store["k"][16] = k[:, k_tag == 16].t().contiguous().clone()
        heads.append(store)
    return heads


@torch.no_grad()
def unpack_layer(packed, keys, values, kept):
    """Rebuild full-size keys and values from `packed`. Evicted tokens keep whatever `keys`/`values` held
    (they are hidden from attention anyway); dropped key channels become zero."""
    keys_out, values_out = keys.clone(), values.clone()
    dtype, dim = keys.dtype, keys.shape[-1]
    for h, store in enumerate(packed):
        n = store["n"]
        v = values.new_zeros((n, dim))
        k = keys.new_zeros((n, dim))
        for bits, block in store["v"].items():
            rows = store["v_tag"] == bits
            v[rows] = block if bits == 16 else _dequantize_rows(block, bits, dim, dtype)
        for bits, block in store["k"].items():
            channels = store["k_tag"] == bits
            data = block if bits == 16 else _dequantize_rows(block, bits, n, dtype)
            k[:, channels] = data.t()
        values_out[0, h][kept[h]] = v
        keys_out[0, h][kept[h]] = k
    return keys_out, values_out


def nbytes(packed):
    """Bytes of every buffer in a packed layer."""
    def size(obj):
        if torch.is_tensor(obj):
            return obj.numel() * obj.element_size()
        if isinstance(obj, dict):
            return sum(size(v) for v in obj.values())
        return 0
    return sum(size(store) for store in packed)
