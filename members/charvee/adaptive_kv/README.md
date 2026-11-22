# AdaptiveKV (Charvee)

Implementation of *Adaptive KV Cache Reuse for Fast Long-Context LLM Serving* (CacheTune, Li et al., [arXiv 2605.24022v2](https://arxiv.org/abs/2605.24022)). No official code exists, so this implementation was written from the paper.

**What it does.** Documents (KV chunks) are encoded once on their own, then reused at any position in later prompts. Reusing them without changes loses the attention *between* chunks. AdaptiveKV recomputes a small share `r` of each chunk's tokens in the new context and loads the cached KV for the rest. A low-frequency FFT score, built offline, picks which tokens to recompute.

> AdaptiveKV cuts **prefill time (TTFT)**. It does **not** shrink the KV cache. Compare it with the team's compression methods on score and TTFT, not on `kv_cache_gb`.

## Files

| File | Contents |
|---|---|
| `adaptive_kv.py` | The method: FFT scoring, cache pool (GPU / CPU / disk), selective-recompute prefill with deferred RoPE, ratio calibration |
| `run_adaptive_kv.py` | LongBench runner (shared data, prompts and scorer) |
| `test_adaptive_kv.py` | Self-tests on a tiny random Llama (CPU, ~10 s, no download) |

## Mapping to the paper

| Paper | Code |
|---|---|
| Eq. 1–4: rFFT, keep lowest α of frequencies, inverse rFFT, K/V norm, per-layer normalisation, layer average | `frequency_scores` |
| Eq. 5: Top-K positions shared across layers | `rank_tokens`, `select_indices` |
| §4.1 recompute in new context; App. B.1 local→global index mapping | `AdaptiveKVEngine._plan`, `prefill` |
| §4.2 load only the (1−r)N complement; store pre-RoPE keys | `ChunkStore.load`, `encode_chunk` hooks on `k_proj`/`v_proj` |
| App. B.3 deferred RoPE at global positions | `prefill` (`_rope(rk, cos[glob], sin[glob])`) |
| App. B.2 transfer stream prefetches layer ℓ+1 during layer ℓ | `prefill` (CUDA only) |
| Eq. 6 / C.4: r0 = t_i / (t_c + t_i), clipped to [r_min, r_max] | `profile_costs`, `analytic_ratio` |
| Algorithm 1: warm-started golden-section search on measured TTFT | `golden_section_search`, `calibrate_ratio` |
| §5.4 ablations: high-frequency, L2, random selection | `--selection high / l2 / random` |
| Baselines: full recompute, full KV reuse | `--mode full_recompute / full_reuse` |

Defaults match the paper: α = 0.5, r = 0.15, r_min = 0.15.

## How to run (lab RTX 5090)

```bash
cd /app/298a-kv-cache-compression
python members/charvee/adaptive_kv/test_adaptive_kv.py          # self-tests

# smoke: 2 records per task on the paper's tasks (samsum, multi_news, musique, 2wikimqa, hotpotqa)
python members/charvee/adaptive_kv/run_adaptive_kv.py --per-task 2

# main comparison
python members/charvee/adaptive_kv/run_adaptive_kv.py --mode full_recompute
python members/charvee/adaptive_kv/run_adaptive_kv.py --mode full_reuse
python members/charvee/adaptive_kv/run_adaptive_kv.py                       # r = 0.15, CPU pool

# selection ablation (Fig. 5) and storage tiers (Fig. 6)
python members/charvee/adaptive_kv/run_adaptive_kv.py --selection random
python members/charvee/adaptive_kv/run_adaptive_kv.py --tier disk --calibrate 10
```

Add `--tasks all` for all 16 LongBench tasks. Results are written to `results/adaptive_kv/longbench/`, with one row per record (score, `ttft_s`, recomputed tokens, `loaded_kv_gb`, offline encode time) and a `.summary.json`.

## Chunking and other choices

- **Chunks.** Multi-hop QA documents are split at each `Passage N:` marker. All other documents are split into blocks of up to 1024 tokens (`--chunk-tokens`). The instruction before the document and the question after it are uncached and always computed.
- **Chunk encoding.** Each chunk is encoded with BOS in front, so the chunk does not act as its own attention sink. The BOS position is then dropped from the stored KV.
- **Exactness checks (in the tests).** With r = 1 the output equals full recompute. With r = 0 and the chunk in its original context, the output also equals full recompute, which checks the pre-RoPE storage, deferred RoPE and fusion. On all three tiers, error falls as r rises.
- **Differences from the paper.** Deferred RoPE runs on the default stream rather than on a third CUDA stream. Attention is plain SDPA over a fused buffer, not a fused kernel. Sliding-window models are not supported. Absolute TTFT will therefore be higher than in the paper, but the comparison across methods within this harness is fair.
