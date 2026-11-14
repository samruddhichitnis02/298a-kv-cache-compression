# ResidualKV on Llama-3.1-8B-Instruct (Charvee)

Draft report section for DATA 298A. All numbers come from the committed files in `results/residualkv/longbench/` and `results/comparison/`. Results were frozen on Nov 13, 2026.

## 1. Method

ResidualKV ([arXiv 2602.08005](https://arxiv.org/abs/2602.08005), formerly DeltaKV; Hao et al.) uses two observations about the KV cache: tokens far apart are often similar, and the difference between similar tokens is small and smooth. It does not throw tokens away. Instead, it splits the cache into two parts:

- **References.** A small set of tokens is stored almost exactly, in FP8. One token in every 10 starts as a reference, and a "dynamic stride" makes references sparser further into the context, so their count grows roughly logarithmically.
- **Residuals.** Every other token is stored as a small 4-bit code. A learned compressor maps the token to a 512-dimensional code, and the stored value is the difference between that code and the codes of its 4 nearest references.

At each attention step, only the top-scoring tokens are rebuilt from their residuals. The rest are skipped. The `--compression-ratio` setting controls this sparse attention: 0.9 means about 10% of the stored tokens are rebuilt in the 27 compressed layers. Five layers (0, 1, 2, 8 and 18) keep every token, in 4-bit KIVI form. Because no token is evicted, a later question can still reach any part of the document.

We ran the authors' code (CURRENTF/DeltaKV, commit `11a837b`, Apache 2.0, kept in `members/charvee/DeltaKV/`) with their released compressor `JitaiHao/Llama-3.1-8B-Instruct-Compressor` and their Llama-3.1-8B settings. The runner is `members/charvee/run_residualkv.py`.

## 2. Setup

| | |
|---|---|
| Model | meta-llama/Llama-3.1-8B-Instruct, bf16 |
| Hardware | NVIDIA RTX 5090 (32 GB), SJSU lab, Docker; PyTorch `sdpa` attention (no flash-attn in the lab image) |
| Data | The team's 800-record LongBench validation set (16 tasks × 50), with the same prompts, chat-template rules and scorer (`harness/longbench.py`) as every other method |
| Settings | Compression ratios 0.75, 0.8, 0.9 and 0.95; 8 sink tokens and the 128 most recent tokens kept raw |
| Statistics | Task-macro average (×100). Each Δ is the difference from the full cache (47.72) with a 95% paired bootstrap interval: records resampled within each task, 2,000 draws. |

**Protocol difference.** The kvpress methods (RestoreKV, KVzip, SnapKV, TriAttention, TurboQuant) compress only the document and add the question afterwards, uncompressed. ResidualKV compresses the whole prompt. Its last 128 tokens, which usually include the question, stay raw, so it chooses tokens with the question in view. This is a small advantage over the kvpress methods.

## 3. Results

### 3.1 Accuracy

| Ratio | Tokens rebuilt per step (paper's "CR") | Score | Δ vs full [95% CI] |
|---|---|---|---|
| Full cache | 100% | 47.72 | |
| 0.75 | 36.7% | 47.83 | +0.12 [−0.80, +1.09] |
| 0.8 | 32.5% | 47.85 | +0.13 [−0.81, +1.14] |
| 0.9 | 24.0% | 47.59 | −0.13 [−1.06, +0.84] |
| 0.95 | 19.8% | 47.38 | −0.33 [−1.31, +0.66] |

"Tokens rebuilt" counts all 32 layers: 5 full layers plus 27 compressed layers at the given keep share.

- No setting is significantly different from the full cache.
- The differences between ratios (47.38 to 47.85) are also within noise, so no ratio is "best".
- By category at ratio 0.8, relative to the full cache:
  - Code and single-document QA are slightly higher: 53.6 vs 51.9, and 44.8 vs 44.0.
  - Summarization is slightly lower: 27.6 vs 28.7. gov_report (−2.5) and samsum (−3.0) lose the most.

### 3.2 Comparison with the paper

The paper (Table III) reports LongBench for Llama-3.1-8B-Instruct: ResidualKV scores **50.3** at 13% storage and 30% compute, against a full-cache average of **50.5**, a gap of −0.2. On our 800 records at ratio 0.8 (32.5% compute), the gap is +0.13, and at 0.9 it is −0.13. **The paper's central claim holds: near-full-cache quality.**

| Category | Paper, ResidualKV | Ours, ResidualKV 0.8 | Ours, full cache |
|---|---|---|---|
| Single-doc QA | 43.2 | 44.8 | 44.0 |
| Multi-doc QA | 47.0 | 42.3 | 41.6 |
| Summarization | 27.9 | 27.6 | 28.7 |
| Few-shot | 69.3 | 70.1 | 70.8 |
| Synthetic | 54.4 | 52.0 | 52.2 |
| Code | 59.8 | 53.6 | 51.9 |
| Average | 50.3 | 47.85 | 47.72 |

Our absolute scores sit about 2.6 points below the paper's on both the full cache and ResidualKV, so the gap comes from the setup, not the method. Three reasons:

- We use an 800-record sample (50 per task), not the full LongBench set.
- Our scorer strips leading whitespace on trec, triviaqa and samsum.
- Prompts longer than 32,000 tokens are truncated.

The team baseline was checked against other published Llama-3.1-8B results in `report/progress_report_3_draft.md`.

### 3.3 Memory: why it stays at about 30%

The measured cache (`kv_cache_gb`, all tensors held in the cache object) is **29.5% of the full cache at every ratio**. The ratio changes only how many stored tokens are rebuilt for attention, never what is stored.

The paper reports 13–16% storage for the same settings. The difference comes from the authors' Hugging Face code, which *simulates* two of its low-precision formats: values are rounded to 4-bit or FP8 and then kept in bf16 tensors. A model of what the cache holds explains 96% of the measured bytes. Each part as a share of the full cache:

| Part | Stored as | Measured | If stored packed, as in the paper |
|---|---|---|---|
| 5 full layers, every token | 4-bit KIVI, simulated, kept in bf16 | 15.6% | about 4.9% |
| Residual codes, 27 layers | 4-bit, really packed | 5.4% | 5.4% |
| References, 27 layers (7.3% of tokens on average) | FP8, simulated, kept in bf16 | 5.4% | 2.7% |
| Reference indices and positions | int64 | 0.8% | 0.8% or less |
| Sink and recent tokens | bf16 | 1.0% | 1.0% |
| Not explained by the model | | 1.2% | |
| **Total** | | **29.5%** | **about 14%** |

The paper's own storage formula, applied to our records, gives **14.2%**. The paper reports 13.4% for this setting.

So ResidualKV's measured memory overstates what the method needs, because two parts are not packed. For RestoreKV 4-bit and TurboQuant, the team counts the packed size. On that same basis ResidualKV uses about 14% memory. That puts it next to RestoreKV 4-bit at 11.4% (47.34). The two are within noise of each other: ResidualKV 0.8 minus RestoreKV 4-bit 0.9 is +0.51 [−0.70, +1.82]. The comparison table keeps the measured 29.6%, and this section gives the packed estimate. Both should be reported.

**How to go lower.** The ratio cannot reduce memory, so lower memory needs different storage settings. In the paper's formula, the 5 full layers (4.9%) and the residual codes (5.4%) dominate. The options are:

- fewer full layers
- 2-bit residual codes (the paper's q₂ = 0.1875, against 0.3125 at 4 bits)
- a smaller latent size than 512
- a sparser reference schedule

None of these were run, so their effect on accuracy is unknown.

### 3.4 Against the other team methods

| Method | Memory | Score | ResidualKV 0.8 minus this method [95% CI] |
|---|---|---|---|
| TurboQuant MSE 3-bit | 20.8% | 47.90 | −0.05 [−1.22, +1.18] |
| TurboQuant MSE 4-bit | 27.0% | 47.57 | +0.28 [−0.76, +1.43] |
| KVzip 0.75 | 26.1% | 47.43 | +0.42 [−1.03, +1.91] |
| RestoreKV 0.75 | 26.1% | 47.31 | +0.54 [−0.71, +1.90] |
| RestoreKV 4-bit 0.9 | 11.4% | 47.34 | +0.51 [−0.70, +1.82] |
| RestoreKV 4-bit 0.95 | 6.4% | 46.63 | +1.22 [−0.51, +2.96] |

ResidualKV ties every method that stays near the full cache. It is clearly better than SnapKV (38.42 at 21%) and TriAttention (37.79 at 21%).

### 3.5 Cost

- Mean time is 3.3 to 3.5 s per record, about 45 minutes for 800 records. That is roughly twice the full cache (1.78 s median) and half of KVzip and RestoreKV (about 7 s).
- Peak GPU memory is 22 GB, against about 24 GB for the kvpress runs. This number includes the 8B model weights and is not a measure of cache savings.
- All 3,200 records finished with no out-of-memory errors and no empty answers.

## 4. Limitations

- **Simulated quantization.** Full layers and references are rounded but stored in bf16, so the measured 29.5% overstates the method's real memory. The packed figure of about 14% is an estimate from the paper's formula and our model of the cache, not a measurement.
- **Protocol difference.** ResidualKV compresses the question together with the document. The kvpress methods add it afterwards, uncompressed.
- **Different code path.** The runner uses the authors' Hugging Face code, not kvpress. Prompts and scoring match the team harness, but the attention kernels and generation loop differ.
- **One memory setting.** Only one storage configuration was tested (the paper's default), so we have no accuracy-versus-memory curve for ResidualKV.
- **Validation set only.** The held-out test split has not been used.

## 5. Plan for DATA 298B

1. Make the storage truly packed (4-bit full layers, FP8 references) and measure the real cache size and GPU memory.
2. Trace a memory curve by varying the number of full layers, the residual bit-width (4 vs 2) and the latent size. Compare it with RestoreKV 4-bit below 15% memory.
3. Run the follow-up question test (`members/anees/run_followup.py`). ResidualKV never evicts tokens, so it should keep its accuracy on a second question.
4. Run the best settings on the held-out test split.

## Reproducing

```bash
python members/charvee/run_residualkv.py --compression-ratio 0.9 --per-task 2   # quick check
python members/charvee/run_residualkv.py --compression-ratio 0.75               # also 0.8, 0.9, 0.95
python results/comparison/compare_methods.py
```
