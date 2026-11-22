"""AdaptiveKV: frequency-guided, hardware-aware non-prefix KV-cache reuse.

Implementation of "Adaptive KV Cache Reuse for Fast Long-Context LLM Serving"
(CacheTune, Li et al., arXiv 2605.24022v2) for Hugging Face decoder models with
RoPE (Llama 3 / 3.1, Mistral, Qwen2, Qwen3).

The method has three parts, each mapped to a section of the paper:

1. Offline index building (Sec. 4.1, App. B.1) -- `AdaptiveKVEngine.encode_chunk`
   Each reusable text chunk is encoded on its own. Its pre-RoPE keys and values
   are kept for every layer, and a token ranking is built from them:
   rFFT along the token axis, keep the lowest `alpha` share of frequencies,
   inverse rFFT back to N positions, score each token by the mean of its K and V
   L2 norms, normalise per layer, average over layers. `frequency_scores` does
   this. The top `r*N` positions of the ranking are recomputed online.

2. Sparse reuse with selective recomputation (Sec. 4.2, App. B.2-B.3) --
   `AdaptiveKVEngine.prefill`. For a prompt assembled from uncached text and
   cached chunks at arbitrary positions, only the uncached tokens and the
   selected chunk tokens carry hidden states through the layers. At each layer
   the selected tokens' K/V are recomputed in the new context, the complement
   is loaded from the cache pool (only those rows: (1-r)*N per chunk), RoPE is
   applied to the reused raw keys at their *global* positions (deferred RoPE),
   and both are scattered into one full K/V buffer that the selected queries
   attend to under a causal mask. On CUDA the next layer's reused K/V are
   prefetched on a separate transfer stream while the current layer computes.

3. Hardware-aware ratio tuning (Sec. 4.3, App. C) -- `profile_costs`,
   `analytic_ratio`, `golden_section_search`, `calibrate_ratio`. The analytic
   balance point r0 = t_i / (t_c + t_i) is clipped to [r_min, r_max] and used
   to warm-start a golden-section search over the measured mean TTFT of a
   calibration set (Algorithm 1).

The cache pool can live on the GPU, in (pinned) CPU memory or on disk
(`numpy.memmap`), matching the storage tiers in the paper.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache

# --------------------------------------------------------------------------- #
# 4.1  Frequency-domain token scoring
# --------------------------------------------------------------------------- #

SELECTION_MODES = ("low", "high", "l2", "random")


def _band_norm(x: torch.Tensor, alpha: float, mode: str) -> torch.Tensor:
    """Per-token L2 norm of `x` ([N, ...]) after keeping one frequency band along the token axis.

    mode="low"  keeps the lowest floor(alpha*F) rFFT bins (the paper's method, Eq. 1-3)
    mode="high" keeps the remaining high-frequency bins (ablation in Sec. 5.4)
    mode="l2"   skips the transform (plain L2-norm ablation in Sec. 5.4)
    """
    n = x.shape[0]
    x = x.reshape(n, -1).float()
    if mode == "l2":
        return x.norm(dim=1)
    spec = torch.fft.rfft(x, dim=0)                       # [F, H*D], F = N//2 + 1
    cutoff = max(1, int(math.floor(alpha * spec.shape[0])))
    if mode == "low":
        spec[cutoff:] = 0
    elif mode == "high":
        spec[:cutoff] = 0
    else:
        raise ValueError(f"unknown band mode {mode!r}")
    recon = torch.fft.irfft(spec, n=n, dim=0)             # back to N token positions
    return recon.norm(dim=1)


def frequency_scores(keys: Sequence[torch.Tensor], values: Sequence[torch.Tensor],
                     alpha: float = 0.5, mode: str = "low", eps: float = 1e-8) -> torch.Tensor:
    """Cross-layer token importance s_i (Eq. 3-4).

    keys / values: one [N, H, D] (or [N, H*D]) pre-RoPE tensor per layer.
    Returns a float32 tensor of N scores on the device of the inputs.
    """
    total = None
    for k, v in zip(keys, values):
        s = 0.5 * (_band_norm(k, alpha, mode) + _band_norm(v, alpha, mode))
        s = s / (s.sum() + eps)                           # per-layer normalisation
        total = s if total is None else total + s
    return total / len(keys)


def rank_tokens(keys, values, alpha=0.5, mode="low", seed=0) -> torch.Tensor:
    """Token positions sorted from most to least important (CPU LongTensor)."""
    n = keys[0].shape[0]
    if mode == "random":
        g = torch.Generator().manual_seed(seed)
        return torch.randperm(n, generator=g)
    scores = frequency_scores(keys, values, alpha=alpha, mode=mode)
    return torch.argsort(scores, descending=True, stable=True).cpu()


def select_indices(ranking: torch.Tensor, ratio: float) -> torch.Tensor:
    """I_freq: the top round(ratio*N) positions of a ranking, sorted by position (Eq. 5)."""
    n_sel = int(round(min(max(ratio, 0.0), 1.0) * len(ranking)))
    return torch.sort(ranking[:n_sel]).values


# --------------------------------------------------------------------------- #
# Cache pool: GPU / CPU / disk tiers, sparse row loading
# --------------------------------------------------------------------------- #

class ChunkStore:
    """Raw (pre-RoPE) K and V of one chunk for all layers: two [L, N, H_kv*D] arrays."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor, tier: str, device,
                 pool_dir: Path | None = None, name: str = "chunk"):
        assert keys.shape == values.shape and keys.dim() == 3
        self.tier = tier
        self.dtype = keys.dtype
        self.shape = tuple(keys.shape)
        self.device = torch.device(device)
        if tier == "gpu":
            self.k, self.v = keys.to(self.device), values.to(self.device)
        elif tier == "cpu":
            pin = self.device.type == "cuda"
            self.k = keys.cpu().pin_memory() if pin else keys.cpu()
            self.v = values.cpu().pin_memory() if pin else values.cpu()
        elif tier == "disk":
            assert pool_dir is not None, "disk tier needs a pool_dir"
            pool_dir.mkdir(parents=True, exist_ok=True)
            self._paths = (pool_dir / f"{name}.k.bin", pool_dir / f"{name}.v.bin")
            self.k = self._write_memmap(self._paths[0], keys)
            self.v = self._write_memmap(self._paths[1], values)
        else:
            raise ValueError(f"unknown tier {tier!r} (gpu, cpu or disk)")

    # bf16 has no numpy dtype, so 16-bit tensors are stored as raw int16 and viewed back.
    def _np_dtype(self):
        return np.int16 if self.dtype.itemsize == 2 else np.int32 if self.dtype.itemsize == 4 else None

    def _write_memmap(self, path: Path, t: torch.Tensor):
        raw = t.detach().cpu().contiguous().view(torch.int16 if t.dtype.itemsize == 2 else torch.int32)
        mm = np.memmap(path, dtype=self._np_dtype(), mode="w+", shape=self.shape)
        mm[:] = raw.numpy()
        mm.flush()
        return np.memmap(path, dtype=self._np_dtype(), mode="r", shape=self.shape)

    @property
    def nbytes_per_row(self) -> int:
        return 2 * self.shape[2] * self.dtype.itemsize    # K and V of one token in one layer

    def load(self, layer: int, idx: torch.Tensor, non_blocking: bool = False):
        """Rows `idx` of layer `layer`, as two [n, H_kv*D] tensors on the compute device."""
        if self.tier == "gpu":
            d = idx.to(self.device)
            return self.k[layer].index_select(0, d), self.v[layer].index_select(0, d)
        if self.tier == "cpu":
            k = self.k[layer].index_select(0, idx)
            v = self.v[layer].index_select(0, idx)
            if non_blocking and self.device.type == "cuda":
                k, v = k.pin_memory(), v.pin_memory()
            return k.to(self.device, non_blocking=non_blocking), v.to(self.device, non_blocking=non_blocking)
        # disk: fancy-index the memmap so only the needed rows are read
        ix = idx.numpy()
        k = torch.from_numpy(np.ascontiguousarray(self.k[layer, ix])).view(self.dtype)
        v = torch.from_numpy(np.ascontiguousarray(self.v[layer, ix])).view(self.dtype)
        if non_blocking and self.device.type == "cuda":
            k, v = k.pin_memory(), v.pin_memory()
        return k.to(self.device, non_blocking=non_blocking), v.to(self.device, non_blocking=non_blocking)

    def delete(self):
        if self.tier == "disk":
            del self.k, self.v
            for p in self._paths:
                p.unlink(missing_ok=True)


@dataclass
class KVChunk:
    """An independently encoded, reusable chunk: its tokens, ranking and stored KV."""
    token_ids: list[int]
    ranking: torch.Tensor
    store: ChunkStore
    encode_seconds: float = 0.0
    rank_seconds: float = 0.0

    def __len__(self):
        return len(self.token_ids)


Segment = Union[KVChunk, Sequence[int]]


@dataclass
class PrefillResult:
    logits: torch.Tensor                 # [vocab] next-token logits at the last prompt position
    cache: DynamicCache                  # full post-RoPE cache, ready for decoding
    seq_len: int
    ttft: float                          # prefill wall time (seconds, device-synchronised)
    n_cached: int = 0                    # tokens that came from KV chunks
    n_recomputed: int = 0                # chunk tokens recomputed in the new context
    loaded_bytes: int = 0                # KV bytes read from the pool
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

def _rope(x, cos, sin):
    """Apply RoPE. x: [B, H, n, D]; cos/sin: [n, D] (already gathered at the positions)."""
    cos, sin = cos[None, None], sin[None, None]
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class AdaptiveKVEngine:
    """Non-prefix KV reuse with frequency-guided selective recomputation.

    model:     a loaded *ForCausalLM with Llama-style decoder layers
    tier:      where chunk KV lives between requests: "gpu", "cpu" or "disk"
    alpha:     low-frequency cutoff share (paper default 0.5)
    selection: "low" (paper), or ablations "high", "l2", "random"
    """

    def __init__(self, model, tier: str = "cpu", alpha: float = 0.5, selection: str = "low",
                 pool_dir: str | os.PathLike | None = None, query_block: int = 2048,
                 pipeline: bool = True, seed: int = 0):
        if selection not in SELECTION_MODES:
            raise ValueError(f"selection must be one of {SELECTION_MODES}")
        self.model = model.eval()
        self.base = model.model
        self.layers = self.base.layers
        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.tier, self.alpha, self.selection = tier, alpha, selection
        self.pool_dir = Path(pool_dir) if pool_dir else Path(".adaptive_kv_pool")
        self.query_block = query_block
        self.pipeline = pipeline and self.device.type == "cuda"
        self.seed = seed
        self._chunks: dict[str, KVChunk] = {}
        types = getattr(cfg, "layer_types", None)
        if types and any(t != "full_attention" for t in types):
            raise NotImplementedError("sliding-window layers are not supported")

    # ---------------- offline: encode a chunk and build its ranking ---------------- #

    @torch.no_grad()
    def encode_chunk(self, token_ids: Sequence[int], prefix_ids: Sequence[int] = ()) -> KVChunk:
        """Encode a chunk on its own and store its pre-RoPE KV plus its frequency ranking.

        `prefix_ids` (e.g. the BOS token) are run in front of the chunk so the chunk
        does not become its own attention sink, then dropped from the stored KV.
        Chunks are memoised by content, so a repeated chunk is encoded only once.
        """
        token_ids = list(token_ids)
        prefix_ids = list(prefix_ids)
        key = hashlib.sha1(np.asarray(prefix_ids + [-1] + token_ids, dtype=np.int64).tobytes()).hexdigest()
        if key in self._chunks:
            return self._chunks[key]

        captured_k, captured_v, hooks = {}, {}, []

        def grab(store, i):
            def hook(_m, _inp, out):
                store[i] = out.detach()
            return hook

        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            # pre-RoPE keys: after k_norm when the model has one (Qwen3), else the projection
            k_mod = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj
            hooks.append(k_mod.register_forward_hook(grab(captured_k, i)))
            hooks.append(attn.v_proj.register_forward_hook(grab(captured_v, i)))

        p = len(prefix_ids)
        ids = torch.tensor([prefix_ids + token_ids], device=self.device)
        _sync(self.device)
        t0 = time.perf_counter()
        try:
            self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
        finally:
            for h in hooks:
                h.remove()
        n = len(token_ids)
        keys = [captured_k[i].reshape(p + n, -1)[p:] for i in range(self.n_layers)]
        vals = [captured_v[i].reshape(p + n, -1)[p:] for i in range(self.n_layers)]
        _sync(self.device)
        t1 = time.perf_counter()
        ranking = rank_tokens(keys, vals, alpha=self.alpha, mode=self.selection, seed=self.seed)
        _sync(self.device)
        t2 = time.perf_counter()

        store = ChunkStore(torch.stack(keys), torch.stack(vals), self.tier, self.device,
                           pool_dir=self.pool_dir, name=key[:16])
        chunk = KVChunk(token_ids, ranking, store, encode_seconds=t1 - t0, rank_seconds=t2 - t1)
        self._chunks[key] = chunk
        return chunk

    def clear_pool(self):
        for c in self._chunks.values():
            c.store.delete()
        self._chunks.clear()

    # ---------------- online: selective recompute + sparse reuse ---------------- #

    def _plan(self, segments: Sequence[Segment], ratio: float):
        """Map each chunk's local recompute/reuse sets to global positions (App. B.1)."""
        ids, active, loads = [], [], []
        pos = 0
        n_cached = n_recomp = 0
        for seg in segments:
            if isinstance(seg, KVChunk):
                n = len(seg)
                sel = select_indices(seg.ranking, ratio)
                mask = torch.ones(n, dtype=torch.bool)
                mask[sel] = False
                reuse_local = torch.nonzero(mask).flatten()
                active.append(sel + pos)
                if len(reuse_local):
                    loads.append((seg.store, reuse_local, reuse_local + pos))
                ids.extend(seg.token_ids)
                n_cached += n
                n_recomp += len(sel)
            else:
                seg = list(seg)
                active.append(torch.arange(pos, pos + len(seg)))
                ids.extend(seg)
                n = len(seg)
            pos += n
        active_pos = torch.cat(active) if active else torch.empty(0, dtype=torch.long)
        if len(active_pos) == 0 or active_pos[-1] != pos - 1:
            # the last prompt token must run through the model to produce the first logits
            active_pos = torch.cat([active_pos, torch.tensor([pos - 1])])
            for i, (store, local, glob) in enumerate(loads):
                if glob[-1] == pos - 1:
                    loads[i] = (store, local[:-1], glob[:-1])
                    n_recomp += 1
        return ids, active_pos, loads, n_cached, n_recomp

    def _fetch(self, loads, layer, non_blocking):
        return [(*store.load(layer, local, non_blocking=non_blocking), glob) for store, local, glob in loads]

    @torch.no_grad()
    def prefill(self, segments: Sequence[Segment], ratio: float = 0.15) -> PrefillResult:
        """Prefill a prompt made of uncached token lists and KVChunks in any order.

        ratio=1.0 recomputes every chunk token (exactly full recompute);
        ratio=0.0 is full KV reuse (only uncached text is computed).
        """
        dev = self.device
        ids, active_pos, loads, n_cached, n_recomp = self._plan(segments, ratio)
        T, A = len(ids), len(active_pos)
        loads = [(s, l, g.to(dev)) for s, l, g in loads]
        active_dev = active_pos.to(dev)
        loaded_bytes = sum(len(l) * s.nbytes_per_row for s, l, _ in loads) * self.n_layers

        _sync(dev)
        t0 = time.perf_counter()

        input_ids = torch.tensor(ids, device=dev)
        h = self.base.embed_tokens(input_ids[active_dev])[None]          # [1, A, hidden]
        cos, sin = self.base.rotary_emb(h, torch.arange(T, device=dev)[None])
        cos, sin = cos[0], sin[0]                                         # [T, D]
        cos_a, sin_a = cos[active_dev], sin[active_dev]
        cache = DynamicCache()
        key_pos = torch.arange(T, device=dev)
        n_rep = self.n_heads // self.n_kv
        hd = self.head_dim

        transfer = torch.cuda.Stream(dev) if self.pipeline and loads else None
        pending = None
        if transfer is not None:
            with torch.cuda.stream(transfer):
                pending = (self._fetch(loads, 0, True), torch.cuda.Event())
                pending[1].record(transfer)

        for li, layer in enumerate(self.layers):
            # ---- reused KV for this layer (prefetched on the transfer stream if pipelined)
            if transfer is not None:
                fetched, ev = pending
                torch.cuda.current_stream(dev).wait_event(ev)
                if li + 1 < self.n_layers:              # prefetch layer li+1 while li computes
                    with torch.cuda.stream(transfer):
                        nxt = self._fetch(loads, li + 1, True)
                        e = torch.cuda.Event()
                        e.record(transfer)
                    pending = (nxt, e)
            else:
                fetched = self._fetch(loads, li, False)

            attn = layer.self_attn
            resid = h
            x = layer.input_layernorm(h)
            q = attn.q_proj(x).view(1, A, -1, hd)
            k = attn.k_proj(x).view(1, A, -1, hd)
            if hasattr(attn, "q_norm"):
                q, k = attn.q_norm(q), attn.k_norm(k)
            v = attn.v_proj(x).view(1, A, -1, hd).transpose(1, 2)
            q = _rope(q.transpose(1, 2), cos_a, sin_a)
            k = _rope(k.transpose(1, 2), cos_a, sin_a)

            # ---- fuse: reused rows (deferred RoPE at global positions) + recomputed rows
            k_full = torch.empty(1, self.n_kv, T, hd, device=dev, dtype=k.dtype)
            v_full = torch.empty_like(k_full)
            for rk, rv, glob in fetched:
                rk = rk.to(k.dtype).view(-1, self.n_kv, hd).transpose(0, 1)[None]
                rv = rv.to(v.dtype).view(-1, self.n_kv, hd).transpose(0, 1)[None]
                k_full[:, :, glob] = _rope(rk, cos[glob], sin[glob])
                v_full[:, :, glob] = rv
            k_full[:, :, active_dev] = k
            v_full[:, :, active_dev] = v
            cache.update(k_full, v_full, li)

            # ---- selected queries attend to the full fused cache (causal by global position)
            kr = k_full.repeat_interleave(n_rep, dim=1) if n_rep > 1 else k_full
            vr = v_full.repeat_interleave(n_rep, dim=1) if n_rep > 1 else v_full
            out = torch.empty_like(q)
            for s in range(0, A, self.query_block):
                e = min(A, s + self.query_block)
                qp = active_dev[s:e]
                kv_end = int(active_pos[e - 1]) + 1          # keys after the last query are masked anyway
                mask = key_pos[None, :kv_end] <= qp[:, None]
                out[:, :, s:e] = F.scaled_dot_product_attention(
                    q[:, :, s:e], kr[:, :, :kv_end], vr[:, :, :kv_end],
                    attn_mask=mask[None, None], scale=attn.scaling)
            out = attn.o_proj(out.transpose(1, 2).reshape(1, A, -1))
            h = resid + out
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        last = self.base.norm(h[:, -1:])
        logits = self.model.lm_head(last)[0, -1].float()
        _sync(dev)
        ttft = time.perf_counter() - t0
        return PrefillResult(logits, cache, T, ttft, n_cached=n_cached, n_recomputed=n_recomp,
                             loaded_bytes=loaded_bytes,
                             stats={"active_tokens": A, "ratio": ratio})

    @torch.no_grad()
    def full_prefill(self, token_ids: Sequence[int]) -> PrefillResult:
        """Baseline: the model's own forward over the whole prompt (full recompute)."""
        dev = self.device
        ids = torch.tensor([list(token_ids)], device=dev)
        _sync(dev)
        t0 = time.perf_counter()
        cache = DynamicCache()
        out = self.model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits = out.logits[0, -1].float()
        _sync(dev)
        return PrefillResult(logits, out.past_key_values, ids.shape[1], time.perf_counter() - t0)

    @torch.no_grad()
    def generate(self, prefill: PrefillResult, max_new_tokens: int, eos_token_ids=()) -> list[int]:
        """Greedy decoding from a prefill result."""
        eos = set(eos_token_ids or ())
        cache, pos = prefill.cache, prefill.seq_len
        tok = int(prefill.logits.argmax())
        out = []
        for _ in range(max_new_tokens):
            if tok in eos:
                break
            out.append(tok)
            if len(out) == max_new_tokens:
                break
            res = self.model(input_ids=torch.tensor([[tok]], device=self.device),
                             position_ids=torch.tensor([[pos]], device=self.device),
                             past_key_values=cache, use_cache=True)
            cache, pos = res.past_key_values, pos + 1
            tok = int(res.logits[0, -1].argmax())
        return out


# --------------------------------------------------------------------------- #
# 4.3  Hardware-aware recomputation ratio
# --------------------------------------------------------------------------- #

@torch.no_grad()
def profile_costs(engine: AdaptiveKVEngine, chunk: KVChunk, repeats: int = 3):
    """Estimate t_c (recompute) and t_i (load) per token per layer, and t_o per layer.

    t_c: full-recompute prefill of the chunk, divided by N*L.
    t_i: loading all N rows of every layer from the pool, divided by N*L.
    t_o: prefill of a 1-token prompt, divided by L (fixed per-layer overhead).
    """
    n, L = len(chunk), engine.n_layers
    all_rows = torch.arange(n)

    def timed(fn):
        fn()                                         # warm-up
        best = float("inf")
        for _ in range(repeats):
            _sync(engine.device)
            t = time.perf_counter()
            fn()
            _sync(engine.device)
            best = min(best, time.perf_counter() - t)
        return best

    t_full = timed(lambda: engine.full_prefill(chunk.token_ids))
    t_load = timed(lambda: [chunk.store.load(li, all_rows) for li in range(L)])
    t_one = timed(lambda: engine.full_prefill(chunk.token_ids[:1]))
    t_o = t_one / L
    t_c = max(t_full - t_one, 1e-9) / (n * L)
    t_i = t_load / (n * L)
    return t_c, t_i, t_o


def analytic_ratio(t_c: float, t_i: float, r_min: float = 0.15, r_max: float = 1.0) -> float:
    """r0 = t_i / (t_c + t_i), clipped to [r_min, r_max] (Eq. 6, C.4-C.5)."""
    r0 = t_i / (t_c + t_i) if (t_c + t_i) > 0 else r_min
    return min(r_max, max(r_min, r0))


def analytic_ttft(r: float, n: int, L: int, t_c: float, t_i: float, t_o: float = 0.0) -> float:
    """Steady-state pipeline estimate T(r) = L*max(r N t_c, (1-r) N t_i) + L t_o (Eq. C.3)."""
    return L * max(r * n * t_c, (1 - r) * n * t_i) + L * t_o


def golden_section_search(f: Callable[[float], float], r_min: float, r_max: float, r0: float,
                          eps: float = 0.02, max_evals: int = 30):
    """Warm-started golden-section search (Algorithm 1). Returns (r*, evaluated {r: f(r)})."""
    seen: dict[float, float] = {}

    def F_(x):
        x = round(x, 6)
        if x not in seen:
            if len(seen) >= max_evals:
                raise RuntimeError("golden-section search exceeded max_evals")
            seen[x] = f(x)
        return seen[x]

    phi = (math.sqrt(5) - 1) / 2
    a, b = r_min, r_max
    r0 = min(b, max(a, r0))
    if r0 <= (a + b) / 2:
        x1, x2 = r0, a + phi * (b - a)
    else:
        x1, x2 = b - phi * (b - a), r0
    if x1 > x2:
        x1, x2 = x2, x1
    if F_(x1) <= F_(x2):
        b = x2
    else:
        a = x1
    x1, x2 = b - phi * (b - a), a + phi * (b - a)
    f1, f2 = F_(x1), F_(x2)
    while b - a >= eps:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - phi * (b - a)
            f1 = F_(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + phi * (b - a)
            f2 = F_(x2)
    return (a + b) / 2, seen


def calibrate_ratio(engine: AdaptiveKVEngine, calibration: Sequence[Sequence[Segment]],
                    r_min: float = 0.15, r_max: float = 1.0, eps: float = 0.02,
                    t_c: float | None = None, t_i: float | None = None, repeats: int = 1):
    """Pick r* minimising mean measured TTFT over calibration prompts (Eq. 7)."""
    if t_c is None or t_i is None:
        chunk = next(s for req in calibration for s in req if isinstance(s, KVChunk))
        t_c, t_i, _ = profile_costs(engine, chunk)
    r0 = analytic_ratio(t_c, t_i, r_min, r_max)

    def mean_ttft(r):
        times = []
        for req in calibration:
            for _ in range(repeats):
                times.append(engine.prefill(req, ratio=r).ttft)
        return sum(times) / len(times)

    engine.prefill(calibration[0], ratio=r0)          # warm-up
    r_star, evals = golden_section_search(mean_ttft, r_min, r_max, r0, eps=eps)
    return {"r_star": r_star, "r0": r0, "t_c": t_c, "t_i": t_i, "evaluations": evals}
