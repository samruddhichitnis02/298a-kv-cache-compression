# TriAttention - Observed Flaws

Every point below is measured from our own LongBench runs on meta-llama/Llama-3.1-8B-Instruct, not taken from the paper. Source numbers are in `results_out/per_task_summary.csv` and `results_out/run_summary.csv`.

These flaws are attributable to TriAttention itself: the press runs the authors' scoring over per-head frequency statistics produced by their calibration (`calibrate_triattention.py` -> `calibration/llama31_8b_stats.pt`), not an approximation. An earlier online-estimate version both invented flaws that do not exist (e.g. it reported hotpotqa collapsing to 16.7 at @0.5, where the calibrated method actually scores 91.7) and exaggerated the real ones, so its flaws could not be trusted. The calibrated results below isolate the method's genuine weaknesses.

Baseline (full cache) macro average: **47.72**. Headline TriAttention run: **TriAttention @0.95** (macro **31.41**, mean KV cache **0.087 GB** vs **1.357 GB** full).

## 1. Accuracy cost at the target budget

At TriAttention @0.95, the macro average falls 16.31 points (34.2% relative) below the full cache. This is a large regression.

## 2. Tasks hurt the most

| Task | Category | Full cache | TriAttention @0.95 | Δ |
|---|---|---|---|---|
| passage_retrieval_en | Synthetic | 100.00 | 36.00 | -64.00 |
| trec | Few-shot | 70.00 | 36.00 | -34.00 |
| multifieldqa_en | Single-Doc QA | 54.85 | 21.83 | -33.02 |
| qasper | Single-Doc QA | 41.63 | 13.30 | -28.33 |
| 2wikimqa | Multi-Doc QA | 45.53 | 19.70 | -25.83 |

## 3. Which capabilities degrade

| Category | Mean Δ vs full cache |
|---|---|
| Synthetic | -31.56 |
| Single-Doc QA | -25.85 |
| Multi-Doc QA | -16.58 |
| Few-shot | -13.35 |
| Summarization | -7.27 |
| Code | -4.30 |

The **Synthetic** tasks lose the most, which points to where TriAttention's pre-RoPE importance score discards information this model actually needed.

## 4. Degradation as the budget shrinks

| Run | Compression | Macro avg | Mean KV (GB) |
|---|---|---|---|
| TriAttention @0.75 | 0.75 | 38.74 | 0.355 |
| TriAttention @0.8 | 0.8 | 37.79 | 0.288 |
| TriAttention @0.9 | 0.9 | 33.10 | 0.154 |
| TriAttention @0.95 | 0.95 | 31.41 | 0.087 |

Accuracy drops more than 10 points below full cache once eviction reaches **0.9** (macro 33.10). Below that budget the method is no longer competitive with the full cache on this model.

![Accuracy vs KV cache](results_out/figures/score_vs_kv_cache.png)

## 5. Why QA degrades: the root cause to target

The tasks that collapse (Single-Doc and Multi-Doc QA) share a property: the answer depends on a few specific evidence tokens scattered through a long document. TriAttention's frequency-domain score ranks a key by how much attention it is expected to receive averaged over future query distances, using the calibrated query centres. That expectation is position/geometry-driven, not query-content-driven, so isolated evidence tokens that matter only for one specific question receive low scores and are evicted. Tasks with redundant or structurally regular signal (code completion, few-shot classification, summarization) survive because the information is spread across many tokens, so losing some is cheap. This points at the concrete lever for improvement: make the score sensitive to the actual question/evidence, not only the calibrated query geometry.

## 6. Measurement scope

Runs cover the LongBench dev prompts under 6K tokens (800 records per run, all 16 tasks). Longer prompts exceed the 32 GB GPU during the uncompressed prefill, so the longest-document examples are not included.
