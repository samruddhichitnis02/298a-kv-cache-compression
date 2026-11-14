# RestoreKV on Llama-3.1-8B-Instruct (Anees)

Draft report section for DATA 298A (SAM-100). All numbers come from committed result files: run `results/comparison/compare_methods.py`, `members/anees/score_followup.py` and `members/anees/make_results_csv.py` to regenerate them. Results were frozen on Nov 13, 2026.

## 1. Method

RestoreKV ([arXiv 2608.01247](https://arxiv.org/abs/2608.01247)) works in two steps:

1. **Eviction.** It drops key-value (KV) pairs with KVzip, which scores each pair by how much it helps the model reconstruct the whole context. That score does not depend on the user's question.
2. **Restoring.** It adds 8 learned "restore" tokens that summarize what was dropped. A small adapter (LoRA) produces them from the evicted pairs.

We use kvpress's `RestoreKVPress` with the authors' released adapter (`higokri/RestoreKV-Llama-3.1-8B-Instruct`), with no retraining. The 8 restore tokens are counted inside the memory budget.

On top of the reproduction, we ran three extensions:

- **N1, quantization at equal memory.** For the same number of cache bytes, is it better to keep few tokens in bf16, or more tokens at 4 or 2 bits? `members/anees/quantized_restorekv_press.py` evicts with RestoreKV, then quantizes the kept cache in the KIVI style with KIVI's group size of 32:
  - Keys: one scale and zero point per channel for every 32 kept tokens.
  - Values: one scale and zero point per token for every 32 channels.

  Scales and zero points are counted in the budget. A kept token costs 160 bytes per head at 4 bits and 96 bytes at 2 bits, against 512 bytes in bf16. So at a 5% memory budget, 4-bit keeps 16.0% of the tokens and 2-bit keeps 26.6%. At 10%, they keep 32.0% and 53.3%.
- **N3, follow-up questions.** Compress a document once, then answer two different questions from the same cache. SnapKV picks tokens using the question, so it should suffer on the second question. RestoreKV and KVzip should not.
- **N2, length.** Does the gap between RestoreKV and KVzip hold as the context grows from 4K to 16K tokens?

## 2. Setup

| | |
|---|---|
| Model | meta-llama/Llama-3.1-8B-Instruct, bf16 |
| Hardware | NVIDIA RTX 5090 (32 GB), SJSU lab, Docker |
| Software | torch 2.7.1+cu128, transformers 5.2.0, kvpress 0.5.5, peft 0.21.2 |
| Data | LongBench validation set: 800 records, 16 tasks × 50. RULER: 13 tasks at 4K (50 per task), 8K and 16K (25 per task). Follow-up: 125 LongBench documents with two or more questions. |
| Harness | Shared team harness `harness/run_longbench.py` and `harness/run_ruler.py`. These give every method the same prompts and the same scorer. |
| Budgets | Compression ratios 0.75, 0.8, 0.9 and 0.95. A ratio is the share of the document's KV pairs removed. |

**Memory** is the computed size of the KV cache: the kept document cache plus the uncompressed question, plus packed codes and metadata for quantized runs. It is reported as a share of the full cache. The full cache averages 1.357 GB on these records. kvpress masks evicted pairs instead of freeing them, so peak GPU memory stays at about 24 GB for every method. These results show accuracy at a given memory size. They do not show real VRAM savings.

**Statistics.** Scores are task-macro averages (×100). Each Δ is the difference from the full cache on the same records, with a 95% paired bootstrap interval: records are resampled within each task, 2,000 draws. An interval that contains 0 means no significant difference.

## 3. Results

### 3.1 RestoreKV against eviction baselines (LongBench)

| Memory | Full cache | RestoreKV | Δ vs full [95% CI] | KVzip | SnapKV |
|---|---|---|---|---|---|
| 26.1% (0.75) | 47.72 | 47.31 | −0.41 [−1.81, +0.98] | 47.43 | 39.40 |
| 21.2% (0.8) | 47.72 | 47.63 | −0.09 [−1.80, +1.56] | 46.00 | 38.42 |
| 11.4% (0.9) | 47.72 | 43.29 | −4.42 [−6.43, −2.38] | 41.61 | 32.64 |
| 6.4% (0.95) | 47.72 | 36.13 | −11.58 [−13.90, −9.14] | 28.32 | 27.24 |

- RestoreKV loses nothing significant down to about 21% memory.
- It beats KVzip, the same eviction without restore tokens, by:
  - +1.63 [+0.28, +3.03] at 0.8
  - +1.68 [+0.11, +3.30] at 0.9
  - +7.81 [+5.87, +9.74] at 0.95
- At 0.75 the two tie (−0.12 [−1.36, +1.13]). So the restore tokens help more as more is evicted, which matches the paper's claim.
- SnapKV trails RestoreKV by 8–11 points at every budget.
- The biggest losses at 6.4% are on exact lookup: passage_retrieval_en falls from 100 to 14, and code completion (lcc) from 54.5 to 37.5. Summarization holds up and is where the restore tokens help most: at 6.4%, RestoreKV keeps multi_news at 24.98 (full 27.01) and qmsum at 25.32 (full 25.06), while KVzip falls to 11.17 and 20.25.

### 3.2 Quantization at equal memory (N1)

| Memory | RestoreKV bf16 | 4-bit | 2-bit | Full cache |
|---|---|---|---|---|
| 6.4% (0.95) | 36.13 | **46.63** | 45.01 | 47.72 |
| 11.4% (0.9) | 43.29 | **47.34** | 44.70 | 47.72 |

Change from bf16 RestoreKV at the same memory:

| | 6.4% | 11.4% |
|---|---|---|
| 4-bit | +10.50 [+8.13, +12.83] | +4.05 [+2.08, +5.94] |
| 2-bit | +8.88 [+6.48, +11.12] | +1.40 [−0.72, +3.41] (not significant) |

- Keeping more tokens at lower precision is clearly better than keeping few tokens in bf16.
- 4-bit at 11.4% memory is within noise of the full cache (−0.38 [−1.78, +0.96]). 4-bit at 6.4% loses only −1.09 [−2.92, +0.74], also not significant.
- **4-bit is the better setting.** 2-bit is never better than 4-bit at the same memory (−1.62 [−3.44, +0.20] at 6.4%).
- A likely reason 2-bit gains little at 11.4%: it keeps 53% of the tokens there, far outside the 2.5–25% keep range the released adapter was trained on. It also rounds every value to one of 4 levels.

In the full comparison of all team methods (`results/comparison/COMPARISON.md`), RestoreKV 4-bit is the best method below 15% memory by a wide margin. The next best at 6.4% is bf16 RestoreKV at 36.13.

### 3.3 Follow-up questions (N3)

Setup: 125 documents (qasper 43, qmsum 34, multifieldqa_en 28, narrativeqa 20) at ratio 0.9. In the "question 1 in the cache" setting, question 1 is appended before compression, as a chat user would ask it. Question 2 then has to use what was kept for question 1.

| Method | Question 1, Δ vs full | Question 2, Δ vs full |
|---|---|---|
| SnapKV | −1.72 [−4.17, +0.93] | **−17.56 [−22.41, −12.74]** |
| KVzip | −2.07 [−6.58, +2.21] | −1.16 [−7.67, +5.78] |
| RestoreKV | −2.49 [−6.53, +1.17] | +1.34 [−3.39, +6.21] |

- SnapKV answers the question it compressed for, but loses 17.6 points on the next one.
- RestoreKV and KVzip choose tokens without the question, so they show no significant loss on either question.
- When only the document is compressed, SnapKV loses about 18–20 points on both questions. RestoreKV again shows no significant loss.
- In a multi-turn chat, where the cache is reused across questions, this is the main practical argument for RestoreKV.

### 3.4 Context length (N2, RULER, ratio 0.95)

| Length | Full cache | RestoreKV | KVzip |
|---|---|---|---|
| 4K | 96.43 | 78.89 | 62.18 |
| 8K | 93.04 | 79.91 | 60.81 |
| 16K | 92.63 | 77.45 | 56.59 |

- RestoreKV stays 17–21 points above KVzip at every length.
- Its score stays roughly flat (77–80), while KVzip's falls by 5.6 points from 4K to 16K.
- Both methods fail on common-word extraction (cwe: 2–9), which needs counts over the whole context and cannot survive keeping 5% of the tokens.
- RestoreKV holds up best on the multi-key and multi-value needle tasks. For example, on niah_multikey_3 at 16K it scores 84, against KVzip's 4.

### 3.5 Cost

| | Median seconds per record | Hours for 800 records |
|---|---|---|
| Full cache | 1.78 | 0.73 |
| SnapKV | 1.8–1.9 | 0.6–0.7 |
| KVzip | 6.5–7.5 | 1.8–2.2 |
| RestoreKV (bf16, 4-bit, 2-bit) | 7.2–7.6 | 2.1–2.25 |

KVzip's scoring step runs an extra reconstruction pass over the context, which makes it about 4× slower than the full cache. RestoreKV and quantization add almost nothing on top of KVzip. All 26 runs finished with no out-of-memory errors.

## 4. Limitations

- **Memory is computed, not freed.** As in kvpress, evicted pairs are masked, and quantized values are rounded in place rather than stored packed. Real VRAM savings and speedups need a packed-cache implementation.
- **The adapter is used outside its training range.** It was trained on bf16 caches with 2.5–25% of tokens kept. Only 4-bit at 6.4% stays inside that range. A quantization-aware adapter could narrow the 2-bit gap.
- **Validation set only.** All LongBench numbers are on the 800-record validation set, and the 800-record test split has not been used. However, 65 of the 250 follow-up questions come from test-split documents. Nothing was tuned on them, but DATA 298B test results should exclude those documents.
- **Scorer fix.** The team scorer strips leading whitespace on trec, triviaqa and samsum, where LongBench's `eval.py` strips only newlines. Without the fix, correct answers that start with " \n" scored 0. Our full-cache score (47.72) is therefore not directly comparable to published LongBench tables.
- **32K was not run.** The length sweep stops at 16K because of the Nov 13 results freeze.

## 5. Plan for DATA 298B

1. Run the 4-bit RestoreKV at 6.4% and 11.4% on the held-out test split, against KVzip, SnapKV and the full cache.
2. Add 32K to the RULER sweep.
3. Store the cache packed so that memory savings are measured on the GPU, not computed.
4. Fine-tune the restore adapter on 4-bit and 2-bit caches (quantization-aware training) and check whether 2-bit closes the gap with 4-bit.

## Reproducing

```bash
python harness/run_longbench.py --press restorekv --compression-ratio 0.95        # also 0.75, 0.8, 0.9; kvzip, snapkv
python harness/run_longbench.py --press restorekv_4bit --compression-ratio 0.95   # also restorekv_2bit, 0.9
python members/anees/run_followup.py --press snapkv --compression-ratio 0.9       # also none, kvzip, restorekv
python harness/run_ruler.py --press restorekv --compression-ratio 0.95 --length 16384
python results/comparison/compare_methods.py
python members/anees/score_followup.py
python members/anees/make_results_csv.py
```

Result files: `results/restorekv/` (LongBench, RULER, follow-up), `results/longbench/` (full-cache baseline), `results/anees.csv` (frozen numbers), `results/comparison/` (all methods).
