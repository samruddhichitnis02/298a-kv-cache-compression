# ResidualKV LongBench results

**Status:** complete (lab RTX 5090, Nov 12–13 2026)  
**Protocol:** shared team 800-record English LongBench dev set  
(`data/processed_longbench/processed_longbench_dev.jsonl`, 16 tasks × 50)  
**Model:** `meta-llama/Llama-3.1-8B-Instruct`  
**Compressor:** `JitaiHao/Llama-3.1-8B-Instruct-Compressor`  
**Runner:** `members/charvee/run_residualkv.py` (HF + sdpa)  
**Baseline:** `results/longbench/Llama-3.1-8B-Instruct_none_0.0` → average **47.72**

Raw files: `results/residualkv/longbench/`

## Overall quality vs baseline

| ratio | approx keep | n | average | Δ vs 47.72 | mean seconds / record | wall time (sum) |
|------:|------------:|--:|--------:|-----------:|----------------------:|----------------:|
| 0.75 | 25% | 800 | **47.83** | **+0.11** | 3.52 s | ~47 min |
| 0.80 | 20% | 800 | **47.85** | **+0.13** | 3.43 s | ~46 min |
| 0.90 | 10% | 800 | **47.59** | **−0.13** | 3.51 s | ~47 min |
| 0.95 | 5% | 800 | **47.38** | **−0.34** | 3.26 s | ~43 min |

Quality stays within about **0.3 points** of the full-cache baseline at every budget. Mild gains at 0.75 / 0.80 are within normal run-to-run noise on this harness.

## Memory (what actually matters for the project)

ResidualKV does **not** shrink the cache to `1 − ratio`. It keeps layers **0, 1, 2, 8, 18** in 4-bit KIVI, plus sink/recent tokens, and stores compressed residuals for dropped tokens. Compare methods on measured `kv_cache_gb`.

| ratio | mean `kv_cache_gb` | median `kv_cache_gb` | mean `full_kv_cache_gb` | mean kept / full |
|------:|-------------------:|---------------------:|------------------------:|-----------------:|
| 0.75 | 0.4017 | 0.3753 | 1.3605 | ~29.5% |
| 0.80 | 0.4016 | 0.3770 | 1.3605 | ~29.5% |
| 0.90 | 0.4015 | 0.3753 | 1.3605 | ~29.5% |
| 0.95 | 0.4015 | 0.3753 | 1.3605 | ~29.5% |

Across these ratios, ResidualKV holds about **0.40 GB** of KV state on average versus **1.36 GB** for a bf16 full cache on the same prompts (~**3.4×** smaller). The ratio knob mainly trades quality, not the logged byte size, because full layers + residuals dominate.

## Per-task scores

| task | baseline | 0.75 | 0.80 | 0.90 | 0.95 |
|---|---:|---:|---:|---:|---:|
| narrativeqa | 35.47 | 35.71 | 35.52 | 35.42 | 34.79 |
| qasper | 41.63 | 42.85 | 42.77 | 42.50 | 43.80 |
| multifieldqa_en | 54.85 | 56.35 | 55.99 | 55.97 | 55.12 |
| hotpotqa | 52.72 | 54.57 | 54.62 | 54.62 | 53.76 |
| 2wikimqa | 45.53 | 43.51 | 43.51 | 43.00 | 44.84 |
| musique | 26.50 | 27.93 | 28.67 | 28.68 | 27.68 |
| gov_report | 33.88 | 30.80 | 31.34 | 30.70 | 30.30 |
| qmsum | 25.06 | 25.00 | 25.55 | 24.58 | 25.37 |
| multi_news | 27.01 | 25.97 | 25.99 | 25.05 | 25.18 |
| trec | 70.00 | 70.00 | 70.00 | 70.00 | 70.00 |
| triviaqa | 93.20 | 94.00 | 94.00 | 94.00 | 94.40 |
| samsum | 49.31 | 45.84 | 46.36 | 46.15 | 46.41 |
| passage_count | 4.50 | 4.00 | 4.00 | 4.00 | 4.00 |
| passage_retrieval_en | 100.00 | 100.00 | 100.00 | 100.00 | 98.00 |
| lcc | 54.46 | 56.43 | 56.35 | 57.31 | 55.62 |
| repobench-p | 49.33 | 52.38 | 50.92 | 49.47 | 48.82 |
| **average** | **47.72** | **47.83** | **47.85** | **47.59** | **47.38** |

## Evaluation method (rubric 4.4)

Accuracy / ROC / MSRE do **not** apply. This is long-context KV-cache compression quality + memory under a fixed protocol:

1. Same 800 prompts, chat-template rules, and `harness/longbench.py` scorer as every other method  
2. Quality: per-task LongBench scores + overall average + Δ vs full-cache **47.72**  
3. Efficiency: per-record `kv_cache_gb`, `full_kv_cache_gb`, `seconds`  
4. Conditions: ratios 0.75 / 0.8 / 0.9 / 0.95 on Llama-3.1-8B-Instruct with the ResidualKV compressor

## Takeaway

All four ratios use the same memory, about **30% of the full cache** (0.40 GB vs 1.36 GB), and all four scores are within noise of the full cache (95% paired bootstrap interval at ratio 0.95: −0.33 [−1.31, +0.66]). The ratio changes which tokens are kept but not the memory, so ResidualKV is one operating point at about 30% memory, not a 5–25% method. At that memory it ties KVzip 0.75 (47.43), RestoreKV 0.75 (47.31) and TurboQuant MSE 4-bit (47.57). The differences between ratios (47.38–47.85) are noise, so no single ratio should be called best.

ResidualKV compresses the question together with the document (see the README), unlike the kvpress methods, which favours it slightly in this comparison.
