"""Shared serving driver: prefill once, optionally pack the cache, answer one or more questions, time it all.

Every method goes through the same three steps, so memory and timing mean the same thing for all of them:

    cache, context_length = prefill(pipe, press, context_ids)     # the press compresses here
    stats = pack_cache(pipe.model, cache, bits=4)                 # optional: store the kept cache packed
    text, timing = answer(pipe, cache, question_ids, context_length, max_new_tokens)

`serve()` wraps the three steps and returns the answers with their timings and the measured cache size.

Packing (the professor's "matched actual byte budgets"). kvpress presses either drop tokens (SnapKV,
TriAttention: the cache tensors really shrink) or only hide them with `masked_key_indices` (KVzip, RestoreKV,
RDKV: the tensors keep their full size). `pack_cache` makes both real: for every layer it gathers the kept
(head, token) pairs, optionally quantizes them KIVI-style to 2, 4 or 8 bits, and frees the original tensors.
Different heads may keep different tokens, so the store is ragged: nothing is padded on disk. At each decoding
step a layer's store is unpacked into a padded tensor just for that attention call and freed again, so the
extra memory is one layer at a time. The padded slots are hidden with the same fake-key mechanism kvpress
already uses. Keys in the cache already carry their rotary position, so compacting them is exact.

`cache_nbytes()` is then the measured size of what is stored. With bits=16 the pack is lossless and the
decoded tokens match the masked run; with 2 or 4 bits it uses the same grouping as
members/anees/quantized_restorekv_press.py (keys per channel over 32 kept tokens, values per token over 32
channels), with scales and zero points stored in the model's dtype and counted.
"""

import contextlib
import time

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer

GROUP_SIZE = 32  # KIVI's default


def sync(device):
    """Wait for queued GPU work, so a timestamp taken next covers it."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


# --------------------------------------------------------------------------------------------
# Packed storage
# --------------------------------------------------------------------------------------------
def pack_codes(codes, bits):
    """uint8 codes below 2**bits, shape (rows, dim) -> (rows, dim * bits / 8) bytes."""
    per_byte = 8 // bits
    if per_byte == 1:
        return codes
    rows, dim = codes.shape
    shifts = torch.arange(per_byte, device=codes.device, dtype=torch.uint8) * bits
    return (codes.view(rows, dim // per_byte, per_byte) << shifts).sum(dim=-1, dtype=torch.uint8)


def unpack_codes(packed, bits):
    per_byte = 8 // bits
    if per_byte == 1:
        return packed
    shifts = torch.arange(per_byte, device=packed.device, dtype=torch.uint8) * bits
    return ((packed.unsqueeze(-1) >> shifts) & (2**bits - 1)).reshape(packed.shape[0], -1)


class PackedLayer(DynamicLayer):
    """One layer's cache: the document part stored packed, tokens added later kept as usual tensors."""

    def __init__(self, keys, values, kept, bits=16, group_size=GROUP_SIZE):
        """keys, values: (1, heads, tokens, dim). kept: bool (heads, tokens), True for pairs to store."""
        super().__init__()
        if keys.shape[0] != 1:
            raise ValueError("PackedLayer supports batch size 1")
        if bits not in (2, 4, 8, 16):
            raise ValueError("bits must be 2, 4, 8 or 16")
        _, self.heads, _, self.dim = keys.shape
        if bits < 16 and (self.dim % group_size or self.dim % (8 // bits)):
            raise ValueError(f"head_dim {self.dim} does not fit group_size {group_size} at {bits} bits")
        self.dtype, self.device = keys.dtype, keys.device
        self.is_initialized = True
        self.bits, self.group_size = bits, group_size
        self.counts = kept.sum(dim=-1)                        # kept pairs per head
        self.width = int(self.counts.max())                   # padded length when unpacked; see set_width
        k, v = keys[0][kept], values[0][kept]                 # (rows, dim), heads one after another
        if bits == 16:
            self.k_store, self.v_store = k.clone(), v.clone()
        else:
            head, rank = self._rows()
            self.k_store = self._quantize(k, self._key_groups(head, rank), key=True)
            self.v_store = self._quantize(v, None, key=False)
        self.keys = keys.new_zeros((1, self.heads, 0, self.dim))
        self.values = values.new_zeros((1, self.heads, 0, self.dim))

    # ---- layout -----------------------------------------------------------------------------
    def _rows(self):
        """For every stored row: its head and its rank among that head's kept tokens."""
        head = torch.repeat_interleave(torch.arange(self.heads, device=self.device), self.counts)
        start = torch.cumsum(self.counts, 0) - self.counts
        rank = torch.arange(head.numel(), device=self.device) - start[head]
        return head, rank

    def _key_groups(self, head, rank):
        """Keys share one scale per channel over each run of `group_size` kept tokens of a head."""
        per_head = torch.div(self.counts + self.group_size - 1, self.group_size, rounding_mode="floor")
        first = torch.cumsum(per_head, 0) - per_head
        return first[head] + torch.div(rank, self.group_size, rounding_mode="floor"), int(per_head.sum())

    # ---- quantization -----------------------------------------------------------------------
    def _quantize(self, x, groups, key):
        levels = 2**self.bits - 1
        x = x.float()
        rows, dim = x.shape
        if key:
            group, n_groups = groups
            index = group.unsqueeze(-1).expand(-1, dim)
            low = x.new_full((n_groups, dim), float("inf")).scatter_reduce(0, index, x, "amin")
            high = x.new_full((n_groups, dim), float("-inf")).scatter_reduce(0, index, x, "amax")
        else:
            grouped = x.view(rows, dim // self.group_size, self.group_size)
            low, high = grouped.amin(dim=-1), grouped.amax(dim=-1)
        scale = (high - low) / levels
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        # Scales and zero points are stored in the model's dtype, so quantize against the stored values.
        low, scale = low.to(self.dtype), scale.to(self.dtype)
        low_x, scale_x = self._expand(low, scale, groups, key, rows)
        codes = torch.round((x - low_x) / scale_x).clamp_(0, levels).to(torch.uint8)
        return {"codes": pack_codes(codes, self.bits), "low": low, "scale": scale}

    def _expand(self, low, scale, groups, key, rows):
        if key:
            return low.float()[groups[0]], scale.float()[groups[0]]
        repeat = lambda t: t.float().repeat_interleave(self.group_size, dim=-1)  # noqa: E731
        return repeat(low), repeat(scale)

    def _dequantize(self, store, groups, key):
        codes = unpack_codes(store["codes"], self.bits).float()
        low, scale = self._expand(store["low"], store["scale"], groups, key, codes.shape[0])
        return (codes * scale + low).to(self.dtype)

    # ---- what the model sees ----------------------------------------------------------------
    def set_width(self, width):
        """Unpack to this many slots per head. All layers use the same width, because the model
        builds one attention mask from the first layer's length."""
        self.width = max(int(width), int(self.counts.max()))

    def padded_slots(self):
        """(batch, head, slot) indices of the empty slots after unpacking; they must not be attended to."""
        slot = torch.arange(self.width, device=self.device).unsqueeze(0).expand(self.heads, -1)
        head, seq = torch.nonzero(slot >= self.counts.unsqueeze(-1), as_tuple=True)
        return torch.zeros_like(head), head, seq

    def unpack(self):
        """Stored pairs as padded (1, heads, width, dim) keys and values, in the model's dtype."""
        head, rank = self._rows()
        if self.bits == 16:
            k, v = self.k_store, self.v_store
        else:
            k = self._dequantize(self.k_store, self._key_groups(head, rank), key=True)
            v = self._dequantize(self.v_store, None, key=False)
        keys = k.new_zeros((self.heads, self.width, self.dim))
        values = v.new_zeros((self.heads, self.width, self.dim))
        keys[head, rank], values[head, rank] = k, v
        return keys.unsqueeze(0), values.unsqueeze(0)

    def update(self, key_states, value_states, cache_kwargs=None):
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        keys, values = self.unpack()
        return torch.cat([keys, self.keys], dim=-2), torch.cat([values, self.values], dim=-2)

    def get_seq_length(self):
        return self.width + self.keys.shape[-2]

    def crop(self, max_length):
        """Drop tokens added after packing, keeping `max_length - width` of them (0 if smaller)."""
        tail = max(int(max_length) - self.width, 0)
        self.keys, self.values = self.keys[..., :tail, :], self.values[..., :tail, :]

    def nbytes(self):
        """Bytes held by this layer: the packed store plus the unpacked tokens added later."""
        def size(store):
            if isinstance(store, dict):
                return sum(t.numel() * t.element_size() for t in store.values())
            return store.numel() * store.element_size()
        tail = sum(t.numel() * t.element_size() for t in (self.keys, self.values))
        return size(self.k_store) + size(self.v_store) + tail


def cache_nbytes(cache):
    """Measured bytes of every tensor the cache holds."""
    total = 0
    for layer in cache.layers:
        if isinstance(layer, PackedLayer):
            total += layer.nbytes()
        elif getattr(layer, "keys", None) is not None:
            total += sum(t.numel() * t.element_size() for t in (layer.keys, layer.values))
    return total


def attention_layers(model):
    language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    return [layer.self_attn for layer in language_model.layers]


@torch.no_grad()
def pack_cache(model, cache, bits=16, group_size=GROUP_SIZE):
    """Replace every layer of `cache` with a PackedLayer holding only the kept pairs.

    Call it after prefill and before decoding. Pairs hidden by a press through
    `module.masked_key_indices` are dropped; presses that already removed tokens keep everything
    that is left. Returns the packed size and what was kept.
    """
    modules = attention_layers(model)
    layers, kept_slots, total_slots = [], 0, 0
    for index, module in enumerate(modules):
        layer = cache.layers[index]
        keys, values = layer.keys, layer.values
        kept = torch.ones(keys.shape[1:3], dtype=torch.bool, device=keys.device)
        masked = getattr(module, "masked_key_indices", None)
        if masked is not None:
            _, head, token = masked
            kept[head.to(kept.device), token.to(kept.device)] = False
        packed = PackedLayer(keys, values, kept, bits, group_size)
        layer.keys = layer.values = None  # free the full-size tensors
        layers.append(packed)
        kept_slots += int(kept.sum())
        total_slots += kept.numel()
    width = max(layer.width for layer in layers)
    for index, (module, layer) in enumerate(zip(modules, layers)):
        layer.set_width(width)
        batch, head, seq = layer.padded_slots()
        module.masked_key_indices = (batch, head, seq) if head.numel() else None
        cache.layers[index] = layer
    return {"bits": bits, "packed_bytes": sum(layer.nbytes() for layer in layers), "kept_slots": kept_slots,
            "total_slots": total_slots, "width": width}


def clear_masks(model):
    """Forget which pairs were hidden, so the next document starts clean."""
    for module in attention_layers(model):
        module.masked_key_indices = None


# --------------------------------------------------------------------------------------------
# Byte budgets
# --------------------------------------------------------------------------------------------
def eviction_ratio_for_budget(memory_budget, bits, num_tokens, model, group_size=GROUP_SIZE):
    """Share of (head, token) pairs to evict so the packed document cache fits in `memory_budget`.

    `memory_budget` is a fraction of the document's full-precision cache. At 16 bits this is
    1 - memory_budget. Below 16 bits each kept pair also pays for its scales and zero points.
    Returns 0 if everything fits (quantization alone is under the budget).
    """
    if not 0 < memory_budget <= 1:
        raise ValueError("memory_budget must be in (0, 1]")
    config = model.config
    module = attention_layers(model)[0]
    head_dim = getattr(module, "head_dim", None) or config.hidden_size // config.num_attention_heads
    element = next(model.parameters()).element_size()
    slots = config.num_hidden_layers * config.num_key_value_heads * num_tokens
    full = slots * 2 * head_dim * element
    if bits == 16:
        return min(max(1 - memory_budget, 0.0), 0.99)
    # codes for the key and the value, the value's scales and zero points, and the pair's share of the key's
    per_slot = 2 * head_dim * bits / 8 + 2 * (head_dim / group_size) * 2 * element
    partial_groups = config.num_hidden_layers * config.num_key_value_heads * head_dim * 2 * element
    kept = (memory_budget * full - partial_groups) / per_slot
    return min(max(1 - kept / slots, 0.0), 0.99)


# --------------------------------------------------------------------------------------------
# Prefill, answer, serve
# --------------------------------------------------------------------------------------------
def _is_prefill_press(press):
    from kvpress.presses.decoding_press import DecodingPress
    return press is not None and not isinstance(press, DecodingPress)


@torch.no_grad()
def prefill(pipe, press, context_ids, chunk=None):
    """Run the document through the model under `press`. Returns (cache, context_length).

    Same as the kvpress pipeline's prefill. With `chunk`, the document is fed in pieces of that many
    tokens so long documents fit in GPU memory; the resulting full cache is the same. Chunking works
    without a press and with presses that score after prefill (KVzip, RestoreKV). Presses that compress
    inside the prefill pass (SnapKV, TriAttention, RDKV) need the whole document in one pass.
    """
    from kvpress import RestoreKVPress
    model = pipe.model
    context_ids = context_ids.to(model.device)
    cache = DynamicCache()
    use_press = _is_prefill_press(press)
    scores_after = use_press and hasattr(press, "_perform_kvzip_compression")
    if chunk and use_press and not scores_after:
        raise ValueError(f"{type(press).__name__} compresses during prefill and cannot be chunked")
    clear_masks(model)
    with press(model) if use_press else contextlib.nullcontext():
        if chunk:
            for start in range(0, context_ids.shape[1], chunk):
                model.model(input_ids=context_ids[:, start:start + chunk], past_key_values=cache)
            if scores_after:
                # KVzip-family presses read the document from the last forward call; give them all of it.
                press._context_ids, press._cache = context_ids, cache
        else:
            model.model(input_ids=context_ids, past_key_values=cache)
    context_length = context_ids.shape[1]
    if isinstance(press, RestoreKVPress):
        context_length = cache.get_seq_length()  # the restore tokens take positions too
    return cache, context_length


@torch.no_grad()
def answer(pipe, cache, question_ids, context_length, max_new_tokens):
    """Greedy answer to one question from `cache`, as in the kvpress pipeline, with timings.

    Returns (text, timing). The tokens this adds to the cache are removed again, so the same cache
    can answer the next question.
    """
    model, device = pipe.model, pipe.model.device
    lengths = [cache.get_seq_length(i) for i in range(len(cache))]
    question_ids = question_ids.to(device)
    position_ids = torch.arange(context_length, context_length + question_ids.shape[1], device=device).unsqueeze(0)
    stop = model.generation_config.eos_token_id
    stop = stop if isinstance(stop, list) else [stop]

    sync(device)
    start = time.perf_counter()
    outputs = model(input_ids=question_ids, past_key_values=cache, position_ids=position_ids, logits_to_keep=1)
    generated = [outputs.logits[0, -1].argmax()]
    sync(device)
    first = time.perf_counter()
    position_ids = position_ids[:, -1:] + 1
    for step in range(max_new_tokens - 1):
        outputs = model(input_ids=generated[-1].unsqueeze(0).unsqueeze(0), past_key_values=cache,
                        position_ids=position_ids + step)
        new_id = outputs.logits[0, -1].argmax()
        generated.append(new_id)
        if new_id.item() in stop:
            break
    sync(device)
    end = time.perf_counter()

    forget_answer(cache, lengths)
    text = str(pipe.tokenizer.decode(torch.stack(generated), skip_special_tokens=True))
    decoded = len(generated) - 1
    return text, {"first_token_s": first - start, "decode_s": end - first, "new_tokens": len(generated),
                  "tokens_per_s": decoded / (end - first) if decoded and end > first else None}


def forget_answer(cache, lengths):
    """Drop the question and answer tokens again, as the kvpress pipeline does between questions."""
    for layer, length in zip(cache.layers, lengths):
        if isinstance(layer, PackedLayer):
            layer.crop(length)
        else:
            layer.keys = layer.keys[:, :, :length]
            layer.values = layer.values[:, :, :length]


def tokenize(pipe, context, questions, answer_prefix=""):
    """The pipeline's own tokenization: chat template, context ids and one id tensor per question.

    `answer_prefix` is one string for all questions, or a list with one per question.
    """
    limit = min(pipe.tokenizer.model_max_length, int(1e10))
    if isinstance(answer_prefix, str):
        return pipe.preprocess(context, questions=questions, answer_prefix=answer_prefix, max_context_length=limit)
    inputs = pipe.preprocess(context, questions=questions[:1], answer_prefix=answer_prefix[0], max_context_length=limit)
    for question, prefix in zip(questions[1:], answer_prefix[1:]):
        # question ids do not depend on the document, so a stand-in document is enough here
        inputs["questions_ids"] += pipe.preprocess("x", questions=[question], answer_prefix=prefix,
                                                   max_context_length=limit)["questions_ids"]
    return inputs


def serve(pipe, press, context, questions, answer_prefix="", max_new_tokens=50, pack_bits=None,
          memory_budget=None, chunk=None):
    """Compress `context` once and answer every question from the same cache.

    pack_bits      None leaves the cache as the press left it; 16, 8, 4 or 2 packs it (see pack_cache).
    memory_budget  with pack_bits: fraction of the full document cache the packed cache may use. The
                   press's eviction ratio is set per document to hit it. Without it the press keeps
                   its own compression_ratio.
    Returns a dict with the answers, per-question timings, and the measured cache size.
    """
    model, device = pipe.model, pipe.model.device
    inputs = tokenize(pipe, context, questions, answer_prefix)
    context_ids = inputs["context_ids"]
    limits = max_new_tokens if isinstance(max_new_tokens, (list, tuple)) else [max_new_tokens] * len(questions)
    requested = getattr(press, "compression_ratio", None)
    if press is not None and pack_bits and memory_budget is not None:
        press.compression_ratio = eviction_ratio_for_budget(memory_budget, pack_bits, context_ids.shape[1], model)
    full_bytes = None
    try:
        sync(device)
        start = time.perf_counter()
        cache, context_length = prefill(pipe, press, context_ids, chunk)
        sync(device)
        prefilled = time.perf_counter()
        full_bytes = _full_bytes(model, context_ids.shape[1])
        packed = pack_cache(model, cache, pack_bits) if pack_bits else None
        sync(device)
        packed_at = time.perf_counter()
        document_bytes = cache_nbytes(cache)
        answers, timings = [], []
        for question_ids, limit in zip(inputs["questions_ids"], limits):
            text, timing = answer(pipe, cache, question_ids, context_length, limit)
            answers.append(text)
            timings.append(timing)
        eviction = getattr(press, "compression_ratio", None)
    finally:
        if requested is not None:
            press.compression_ratio = requested
        clear_masks(model)
    first = timings[0]
    return {
        "answers": answers,
        "questions": timings,
        "context_tokens": int(context_ids.shape[1]),
        "question_tokens": [int(q.shape[1]) for q in inputs["questions_ids"]],
        "prefill_s": prefilled - start,                 # document forward pass plus the press's own work
        "pack_s": packed_at - prefilled,
        "ttft_s": (packed_at - start) + first["first_token_s"],
        "eviction_ratio": eviction,
        "packed": packed,
        "document_cache_bytes": document_bytes,          # measured: tensors held for the document
        "full_document_cache_bytes": full_bytes,         # the same document uncompressed
    }


def _full_bytes(model, num_tokens):
    config = model.config
    module = attention_layers(model)[0]
    head_dim = getattr(module, "head_dim", None) or config.hidden_size // config.num_attention_heads
    element = next(model.parameters()).element_size()
    return 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * element * num_tokens


def median_timing(runs):
    """Median of each timing over repeated `serve` results for the same input."""
    def median(values):
        values = sorted(v for v in values if v is not None)
        if not values:
            return None
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    merged = {key: median([r[key] for r in runs]) for key in ("prefill_s", "pack_s", "ttft_s")}
    first = [r["questions"][0] for r in runs]
    for key in ("first_token_s", "decode_s", "tokens_per_s"):
        merged[key] = median([f[key] for f in first])
    merged["new_tokens"] = first[0]["new_tokens"]
    merged["repeats"] = len(runs)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in merged.items()}
