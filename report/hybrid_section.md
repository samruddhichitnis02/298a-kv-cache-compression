# Hybrid eviction + quantization at matched byte budgets

Draft section for the final report. It answers the question raised in review: at the same actual memory, is it better to drop tokens, to store them at lower precision, or to do both? All numbers come from `results/comparison/HYBRID.md`, which `results/comparison/hybrid_tables.py` regenerates from the result files in `results/hybrid/longbench`. The runs were done by Manjot on an RTX 5090.

## Setup

- **One budget.** Every method gets 10% of the full document cache, in bytes that are actually stored. The kept keys and values are written to packed buffers and the originals are freed, so the size is counted from the stored tensors. It is not calculated from a formula.
- **Three ways to spend it.** Eviction only drops 90% of the cached pairs and keeps the rest at 16 bits. Eviction plus 4-bit drops 68% and stores the rest at 4 bits. Eviction plus 2-bit drops 47% and stores the rest at 2 bits. The drop rate is solved per document so that the packed size lands on the budget, including the scales and zero points that quantization needs.
- **Quantization.** Keys are quantized per channel and values per token, in groups of 32, as in KIVI. The question and the answer stay at full precision.
- **Methods.** KVzip, RestoreKV, SnapKV and TriAttention choose which pairs to drop. RDKV gives each pair its own bit-width, so its hybrid is its normal mode, and its eviction-only version allows bit-widths 0 and 16 only.
- **Data.** 400 LongBench validation records, 25 per task, the same records in every run. The uncompressed model scores 48.66 on these records. Its score on all 800 is 47.72, so numbers here should be compared with 48.66.
- **What was measured.** The measured cache is between 9.99% and 10.00% of the full cache in all twelve budgeted runs, and 10.32% for RDKV with mixed bit-widths. No run had an out-of-memory record.

## Finding 1: at the same bytes, the hybrid beats eviction only for every method

| Method | Eviction only | Evict + 4-bit | Evict + 2-bit | Best hybrid minus eviction only |
|---|---|---|---|---|
| KVzip | 43.53 | 50.00 | 45.85 | +6.47 (+3.51 to +9.36) |
| RestoreKV | 44.57 | 48.74 | 47.25 | +4.18 (+1.51 to +6.86) |
| SnapKV | 34.69 | 41.90 | 43.81 | +9.13 (+5.95 to +12.29) |
| TriAttention | 34.98 | 42.30 | 41.87 | +7.31 (+4.69 to +10.03) |
| RDKV | 35.90 | 43.00 (mixed bit-widths) | | +7.11 (+4.75 to +9.81) |

Intervals are 95%, paired over the 400 records.

- **The gain is 4 to 9 points and the interval excludes zero for all five methods.** Keeping more pairs at lower precision is a better use of the same bytes than keeping few pairs at full precision.
- **The weaker the method's choice of what to drop, the more it gains.** SnapKV, TriAttention and RDKV gain 7 to 9 points; KVzip and RestoreKV, which choose better, gain 4 to 6.
- **4 bits or 2 bits is too close to call.** 4-bit is ahead for KVzip and RestoreKV and 2-bit for SnapKV, but for SnapKV and TriAttention the two hybrids are within 2 points of each other.

## Finding 2: two hybrids match the uncompressed model at 10% of its memory

| Setting | Cache size | Score | Minus uncompressed |
|---|---|---|---|
| RestoreKV, evict + 4-bit | 10.0% (measured) | 48.74 | +0.09 (-1.96 to +2.05) |
| KVzip, evict + 4-bit | 10.0% (measured) | 50.00 | +1.35 (-0.98 to +3.62) |
| TurboQuant, MSE 2.5-bit | 18.5% (calculated) | 48.68 | +0.03 (-2.06 to +2.10) |
| 2-bit, nothing dropped | 18.8% (measured) | 46.93 | -1.73 (-3.92 to +0.29) |
| TurboQuant, MSE 3-bit | 20.8% (calculated) | 49.29 | +0.64 (-0.67 to +1.94) |
| ResidualKV, ratio 0.9 | 29.6% (calculated) | 48.74 | +0.08 (-1.36 to +1.51) |
| 4-bit, nothing dropped | 31.3% (measured) | 48.61 | -0.04 (-1.31 to +1.19) |
| Uncompressed | 100% | 48.66 | |

The TurboQuant and ResidualKV rows are earlier runs restricted to the same 400 records. They are not at the 10% budget; they show where methods that quantize without dropping sit.

- **KVzip and RestoreKV with 4-bit are level with the uncompressed model at a tenth of its memory.** Neither difference is distinguishable from zero. KVzip's 50.00 is not a gain over the full cache: the interval runs from -0.98 to +3.62.
- **Quantization alone needs two to three times the memory for the same score.** 4-bit with nothing dropped loses nothing, but costs 31.3%. TurboQuant reaches the same score at 18.5%. The hybrids reach it at 10%.
- **Quantization alone cannot go below about 19%** with this scheme, because 2 bits is the floor and it already costs 1.7 points. Dropping pairs is the only way further down, which is why the two are worth combining.
- **SnapKV, TriAttention and RDKV stay 5 to 6 points below the uncompressed model** even as hybrids. Quantization gives back part of what their eviction loses, not all of it.

## Finding 3: the hybrid repairs the task types that eviction breaks

Score by task type at 10% measured memory: eviction only, then the best hybrid.

| Method | Single-doc QA | Multi-doc QA | Summarization | Few-shot | Synthetic | Code |
|---|---|---|---|---|---|---|
| Uncompressed | 45.7 | 46.1 | 28.3 | 70.0 | 50.5 | 53.5 |
| KVzip | 42.7 → 47.3 | 45.3 → 49.1 | 27.9 → 28.7 | 61.7 → 69.2 | 26.0 → 50.0 | 55.7 → 58.6 |
| RestoreKV | 47.5 → 45.8 | 39.3 → 45.4 | 27.8 → 29.0 | 68.0 → 68.0 | 26.0 → 50.9 | 56.5 → 56.7 |
| SnapKV | 20.4 → 36.2 | 27.9 → 41.5 | 21.1 → 27.3 | 62.5 → 69.3 | 26.1 → 34.2 | 53.6 → 54.7 |
| TriAttention | 20.2 → 32.2 | 30.5 → 38.2 | 22.7 → 26.2 | 60.5 → 62.1 | 29.0 → 45.8 | 50.2 → 54.5 |
| RDKV | 23.0 → 31.8 | 32.7 → 41.0 | 21.8 → 25.6 | 62.7 → 68.6 | 22.0 → 36.7 | 54.7 → 56.7 |

- **The synthetic tasks (passage retrieval and counting) are where eviction fails and the hybrid recovers.** With eviction only, every method scores 22 to 29 against 50.5 uncompressed. With the hybrid, KVzip and RestoreKV are back at 50. These tasks need many passages to stay in the cache, and the hybrid keeps three to five times as many pairs.
- **Code and summarization barely move.** They were already close to the uncompressed score with eviction only.
- **This agrees with the task-adaptive section,** which found from separate runs that retrieval collapses under eviction and survives quantization.

## Limits

- 400 records, 25 per task. Intervals on the macro-average are about ±2 to ±3 points, so differences smaller than that are not established. That covers 4-bit against 2-bit, and KVzip against RestoreKV.
- One budget, 10%. We did not test whether the hybrid's advantage holds at 5% or 20%.
- "Measured" means the bytes of the stored cache. While the model decodes, the packed cache is unpacked for attention, so peak GPU memory during decoding is higher than the stored size. The serving section reports peak memory and the unpacking cost.
- The TurboQuant and ResidualKV sizes are calculated, not measured from packed buffers, and come from earlier runs.
- TriAttention uses the first calibration. Its rows will change when the rerun with the corrected calibration is in.
