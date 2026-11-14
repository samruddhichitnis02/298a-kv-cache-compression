# ResidualKV (Charvee)

[ResidualKV](https://arxiv.org/abs/2602.08005), called DeltaKV in the first arXiv version, keeps a few layers whole and, in the other layers, replaces dropped tokens with small compressed residuals against similar tokens that are kept. It uses the authors' code ([CURRENTF/DeltaKV](https://github.com/CURRENTF/DeltaKV), commit `11a837b`) and their released Llama-3.1-8B-Instruct compressor, `JitaiHao/Llama-3.1-8B-Instruct-Compressor`.

Settings follow Charvee's configs. The runner `run_residualkv.py` was set up by Anees so that ResidualKV uses the same 800 dev records, prompts, scorer and results format as the other methods.

Authors' DeltaKV code is vendored at `members/charvee/DeltaKV/` (no separate clone required). Lab zip layout may still use `/app/DeltaKV`; override with `--deltakv-repo` if needed.

**Full LongBench results (lab complete):** see [`RESULTS.md`](RESULTS.md). Report section: [`docs/CHARVEE_REPORT_CHAPTER.md`](docs/CHARVEE_REPORT_CHAPTER.md).

## Results summary (800 records, RTX 5090)

Baseline full cache: **47.72**. Files under `results/residualkv/longbench/`.

| ratio | approx keep | average | Δ vs baseline | mean `kv_cache_gb` | mean `full_kv_cache_gb` |
|------:|------------:|--------:|--------------:|-------------------:|------------------------:|
| 0.75 | 25% | **47.83** | +0.11 | 0.4017 | 1.3605 |
| 0.80 | 20% | **47.85** | +0.13 | 0.4016 | 1.3605 |
| 0.90 | 10% | **47.59** | −0.13 | 0.4015 | 1.3605 |
| 0.95 | 5% | **47.38** | −0.34 | 0.4015 | 1.3605 |

Quality stays within ~0.3 of baseline. Logged KV is ~**0.40 GB** vs ~**1.36 GB** full (~3.4× smaller). The ratio is not an exact memory percentage (full layers + sink/recent + residuals remain).

## How to run (lab RTX 5090)

```bash
cd /app/298a-kv-cache-compression
pip install --break-system-packages transformers==5.2.0 rouge rapidfuzz loguru accelerate
# hf auth login   # gated Llama access

# 1. Quick check, 2 records per task (~10 minutes)
python members/charvee/run_residualkv.py --compression-ratio 0.9 --per-task 2

# 2. Full runs
python members/charvee/run_residualkv.py --compression-ratio 0.9
python members/charvee/run_residualkv.py --compression-ratio 0.75
python members/charvee/run_residualkv.py --compression-ratio 0.8
python members/charvee/run_residualkv.py --compression-ratio 0.95
```

Reruns skip `record_id`s already in the jsonl.

## Memory

ResidualKV keeps layers 0, 1, 2, 8 and 18 whole (4-bit), plus 8 sink and 128 recent tokens, and stores compressed residuals for dropped tokens. So `--compression-ratio 0.9` is **not** a 10% cache. Each record logs the bytes actually held in its cache (`kv_cache_gb`) and the bf16 size of a full cache for the same tokens (`full_kv_cache_gb`). Compare methods on `kv_cache_gb`.

## Differences from the shared harness

- ResidualKV compresses the whole prompt, keeping the last 128 tokens, which usually include the question. kvpress methods compress only the document, so ResidualKV chooses what to keep with the question in view. Read the comparison with this in mind.
- The attention kernel is PyTorch's `sdpa`, because the lab image has no flash-attn.
