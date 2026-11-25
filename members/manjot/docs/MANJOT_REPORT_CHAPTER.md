# RDKV on Llama-3.1-8B-Instruct (Manjot)

Draft report section for DATA 298A. All numbers come from the committed files in `results/rdkv/longbench/` and `results/comparison/`. The runs were committed on Nov 19, 2026 (PR #102) and were checked by rescoring the saved predictions.

## 1. Method

RDKV ([arXiv 2605.08317](https://arxiv.org/abs/2605.08317); Zhang, Guo, Benini and Li, 2026) treats eviction and quantization as one decision. Every unit of the cache gets a bit-width from {0, 2, 4, 8, 16}. Zero bits means the unit is evicted and 16 bits means it is kept in full precision. The bit-widths are chosen by rate-distortion allocation: spend bits where an error would hurt attention most, under a fixed bit budget.

- **Values are allocated per token.** A token's weight is the attention it receives from the last 32 context tokens, smoothed over 5 neighbours.
- **Keys are allocated per channel.** A channel's weight is the product of the query norm and the key norm in that channel.
- **The budget is split equally** between keys and values and is the same for every layer and head.

No official code is public, so `members/manjot/rdkv_press.py` is our own implementation as a kvpress press. It runs once at the end of prefill, per layer and per KV head:

1. Score tokens and channels.
2. Allocate bits to the values of each token. Tokens that get 0 bits are evicted.
3. Allocate bits to the key channels of the tokens that remain. Channels that get 0 bits are removed.

The allocation is solved by Lagrangian relaxation with bisection, for all heads in parallel.

**Differences from the paper.**

- The paper estimates the error of each bit-width offline on 32 calibration sequences. We compute the exact quantization error of each unit online, so no calibration set is needed.
- The values of the last 32 context tokens always stay at full precision and are charged to the budget. Their keys still go through channel allocation.
- Quantization is simulated (quantize, then dequantize) and eviction is masked. GPU memory does not drop, so memory is reported as a computed size, as for the other kvpress methods.
- We did not implement the paper's packed storage format or its GPU kernels, so we report no speed numbers.

## 2. Setup

| | |
|---|---|
| Model | meta-llama/Llama-3.1-8B-Instruct, bf16 |
| Hardware | One NVIDIA RTX 5090 (32 GB) on RunPod; transformers 5.2.0, kvpress 0.5.5 |
| Data | The team's 800-record LongBench validation set (16 tasks × 50), with the shared prompts and scorer (`harness/longbench.py`) |
| Settings | Compression ratios 0.75, 0.8, 0.9 and 0.95, no prompt-token cap. The document is compressed without seeing the question, which is added afterwards uncompressed. |
| Statistics | Task-macro average (×100). Each Δ is the difference from the full cache (47.72) with a 95% paired bootstrap interval: records resampled within each task, 2,000 draws. |

All four runs completed 800 of 800 records with no out-of-memory failures, in about 0.63 hours each. Peak GPU memory was 24.2 GB, about the same as the full-cache baseline (24.1 GB), because the compression is simulated.

## 3. Results

### 3.1 Accuracy

| Ratio | Memory | Tokens kept | Score | Δ vs full [95% CI] |
|---|---|---|---|---|
| Full cache | 100% | 100% | 47.72 | |
| 0.75 | 26.7% | 76.1% | 47.16 | −0.55 [−1.57, +0.47] |
| 0.8 | 21.7% | 64.7% | 46.92 | −0.80 [−2.10, +0.48] |
| 0.9 | 11.6% | 35.0% | 42.58 | −5.14 [−6.83, −3.42] |
| 0.95 | 6.6% | 17.4% | 38.03 | −9.69 [−11.76, −7.66] |

- **At 22% memory and above, RDKV is not significantly different from the full cache.**
- **Below that it degrades quickly:** 5 points at 12% memory and almost 10 points at 7%.
- **RDKV keeps far more tokens than an eviction method at the same memory.** At ratio 0.9 it keeps 35% of the tokens in about 10% of the bits, by storing most of them at low precision.

### 3.2 Memory

Memory is the computed cache size as a share of the full cache: the compressed document plus the uncompressed question.

| Ratio | Document payload | Document with quantizer metadata | Whole cache, with question |
|---|---|---|---|
| 0.75 | 24.6% | 25.2% | 26.7% |
| 0.8 | 19.7% | 20.2% | 21.7% |
| 0.9 | 9.8% | 10.1% | 11.6% |
| 0.95 | 4.9% | 5.1% | 6.6% |

- The allocator meets its bit budget: the document payload is within 0.4 points of the target at every ratio.
- Quantizer metadata (a scale and zero-point per quantized token or channel, plus keep bitmaps) adds 0.2 to 0.6 points.
- The last column is the figure used in the team comparison. It is 0.2 to 0.6 points above the other kvpress methods at the same ratio, because their sizes carry no such metadata.

### 3.3 By task

| Task | Full cache | 0.75 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|
| narrativeqa | 35.5 | 36.6 | 35.4 | 32.8 | 29.0 |
| qasper | 41.6 | 37.6 | 35.2 | 32.5 | 30.0 |
| multifieldqa_en | 54.9 | 47.7 | 46.8 | 35.9 | 23.4 |
| hotpotqa | 52.7 | 53.8 | 55.0 | 44.1 | 43.6 |
| 2wikimqa | 45.5 | 46.5 | 47.3 | 36.5 | 25.3 |
| musique | 26.5 | 25.1 | 28.0 | 27.0 | 19.3 |
| gov_report | 33.9 | 32.1 | 31.0 | 29.1 | 27.2 |
| qmsum | 25.1 | 24.9 | 25.8 | 23.2 | 21.1 |
| multi_news | 27.0 | 26.6 | 26.4 | 23.8 | 22.1 |
| trec | 70.0 | 72.0 | 70.0 | 68.0 | 62.0 |
| triviaqa | 93.2 | 91.2 | 91.2 | 93.4 | 92.4 |
| samsum | 49.3 | 48.5 | 48.5 | 48.5 | 48.4 |
| passage_count | 4.5 | 4.8 | 7.1 | 5.1 | 9.3 |
| passage_retrieval_en | 100.0 | 100.0 | 94.0 | 76.0 | 56.0 |
| lcc | 54.5 | 55.1 | 57.6 | 58.0 | 54.9 |
| repobench-p | 49.3 | 51.9 | 51.5 | 47.3 | 44.4 |

- **Few-shot and code tasks hold up.** samsum and lcc stay within 1 point of the full cache, or above it, at every ratio, and triviaqa within 2 points.
- **Single-document QA is the weak point.** qasper and multifieldqa_en already lose 4 and 7 points at ratio 0.75, where the average loss is only 0.55. At ratio 0.95, multifieldqa_en loses 31 points.
- **Passage retrieval falls sharply at high ratios:** from 100 to 76 at ratio 0.9 and to 56 at ratio 0.95.

### 3.4 Comparison with the other methods

Scores at the same compression ratio. RDKV's memory is 0.2 to 0.6 points higher than the others at each ratio (Section 3.2).

| Ratio | RDKV | RestoreKV 4-bit | RestoreKV | KVzip | SnapKV |
|---|---|---|---|---|---|
| 0.75 | 47.16 | n/a | 47.31 | 47.43 | 39.40 |
| 0.8 | 46.92 | n/a | 47.63 | 46.00 | 38.42 |
| 0.9 | 42.58 | 47.34 | 43.29 | 41.61 | 32.64 |
| 0.95 | 38.03 | 46.63 | 36.13 | 28.32 | 27.24 |

- **RDKV beats SnapKV by 8 to 11 points at every ratio.** Both score tokens by the attention they receive from the end of the document, so the gain most likely comes from the bit allocation.
- **RDKV is level with eviction-only RestoreKV and KVzip** down to 12% memory, and ahead of both at 7%.
- **RestoreKV with 4-bit quantization is well ahead** at 12% and 7% memory, by 4.8 and 8.6 points. Both methods combine eviction with quantization, but they choose which tokens to keep differently: RDKV uses attention from the last 32 context tokens.
- **By task type,** RDKV is the best or tied-best method on few-shot tasks at all four ratios (`results/comparison/TASK_ADAPTIVE.md`, added in PR #104).

### 3.5 Comparison with the paper

The paper's Table 1 reports LongBench for Llama-3.1-8B-Instruct at budgets of 64 to 1,024 tokens per layer. RDKV scores 45.97 at the smallest budget and 49.47 at the largest, against a full cache of 49.82. SnapKV scores 38.95 and 48.72 at the same two budgets.

- **The ordering reproduces.** RDKV is ahead of SnapKV at every budget in the paper and at every ratio in our runs.
- **The size of the loss does not.** The paper loses 0.35 points at its largest budget and 3.85 at its smallest. We lose 5.14 points at 12% memory.

The budgets are defined differently (a fixed token count per layer in the paper, a share of the prompt in ours), so the two are not directly comparable. Three differences in our setup could also explain a larger loss:

- Our error estimates are computed online, not calibrated offline as in the paper.
- Our pipeline compresses the document before the question arrives. Token weights therefore come from the end of the document, not from the question.
- Ours is a re-implementation with no official code to check against.

We have not tested which of these matters most.

## 4. Limits

- **Simulated compression.** Memory is computed, not measured on the GPU, and there are no speed measurements.
- **Re-implementation.** No official code exists, so faithfulness to the paper is checked only by unit tests of the allocator and by the ordering in Section 3.5.
- **Different machine.** The runs used a rented RTX 5090, not the lab machine. Scores should not depend on the machine, but the times in Section 2 should not be compared with other methods' times.
- **No code commit recorded.** The result files do not record which commit of `rdkv_press.py` produced them.
- **Validation set only.** The 800-record test split has not been used.

## 5. Reproducing

```bash
python -m pytest -q members/manjot/test_rdkv.py                         # 9 tests, CPU
python harness/run_longbench.py --press rdkv --compression-ratio 0.9    # also 0.75, 0.8, 0.95
python members/manjot/compare.py results/rdkv/longbench/Llama-3.1-8B-Instruct_rdkv_0.9.jsonl
python results/comparison/compare_methods.py
```
