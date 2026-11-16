# TriAttention on Llama-3.1-8B-Instruct (Samruddhi)

Draft report section for DATA 298A. The numbers come from the committed files in `results/triattention/longbench/` and `results/comparison/`. Samruddhi's own write-ups are in `members/samruddhi/` (README, RESULTS, COMPARISON, FLAWS). **A re-run with a corrected calibration is pending; see §4.**

## 1. Method

TriAttention ([arXiv 2604.04921](https://arxiv.org/abs/2604.04921), Mao et al., ICML 2026) decides which cached keys to keep without looking at attention scores. Its starting point is that, before rotary position encoding (RoPE) is applied, a head's query vectors cluster tightly around a fixed center.

- **Distance preference from calibration.** Because the queries cluster, how much a head attends to a key depends mostly on the distance between them. A trigonometric series built from the query center predicts this preference.
- **Key score.** Each key's score combines that series (S_trig), evaluated at a range of future distances spaced 1, 2, 4 and so on, with a norm term (S_norm). The weight between the two depends on how tightly a head's queries cluster.
- **Offline calibration.** The query centers come from one offline pass over calibration text.
- **Eviction.** The top-scoring keys are kept, plus the first few "sink" tokens and the most recent tokens.

Unlike SnapKV, the score does not depend on the current question.

The paper evaluates mainly long reasoning (AIME, MATH 500) with token budgets of 512 to 4,096, pruning every 128 generated tokens. LongBench appears only in its Appendix F, on Qwen3-8B at a 50% budget.

Samruddhi implemented the method as a kvpress scorer, `members/samruddhi/triattention_press.py`. It is based on the authors' `pruning_utils.py`:

- it inverts RoPE on the cached keys
- it uses complex frequency pairs
- it sums the series over geometric offsets up to 65,536 and averages them
- it applies the (1 − R) norm weighting

The query statistics come from `calibrate_triattention.py`.

## 2. Setup

| | |
|---|---|
| Model | meta-llama/Llama-3.1-8B-Instruct, bf16 |
| Hardware | RTX 5090 under WSL2 with FlashAttention. The native Windows PyTorch build has no flash attention and ran out of memory on 32K prompts. |
| Data | The team's 800-record LongBench validation set, same prompts and scorer as every method (`harness/run_longbench.py`) |
| Settings | Compression ratios 0.75, 0.8, 0.9 and 0.95. The document is compressed once after reading; the question is added afterwards, uncompressed. 4 sink tokens and 64 recent tokens are always kept. |
| Statistics | Task-macro average (×100). Each Δ is a 95% paired bootstrap interval: records resampled within each task, 2,000 draws. |

## 3. Results (Nov 10 calibration)

| Ratio | Memory | Score | Δ vs full cache (47.72) | Δ vs SnapKV at the same ratio |
|---|---|---|---|---|
| 0.75 | 26.1% | 38.74 | −8.97 [−10.96, −7.10] | −0.66 [−2.79, +1.23] |
| 0.8 | 21.2% | 37.79 | −9.92 [−11.96, −7.93] | −0.63 [−2.68, +1.40] |
| 0.9 | 11.4% | 33.10 | −14.62 [−17.01, −12.31] | +0.46 [−1.48, +2.52] |
| 0.95 | 6.4% | 31.41 | −16.31 [−18.62, −14.05] | **+4.17 [+2.16, +6.17]** |

- **Large loss even at the lightest setting.** TriAttention loses 9 points at 26% memory, where KVzip, RestoreKV and ResidualKV lose nothing significant.
- **Ties SnapKV except at the smallest budget.** At 0.75 to 0.9 it is within noise of SnapKV. It beats SnapKV only at 0.95, by +4.2.
- **Biggest losses are on retrieval-heavy tasks.** At 0.95: passage_retrieval_en falls from 100 to 36, multifieldqa_en from 54.9 to 21.8, qasper from 41.6 to 13.3, and trec from 70 to 36.
- **Tasks that hold up.** triviaqa (92.2), samsum (44.3), lcc (47.2) and repobench-p (48.0) stay close to the full cache, as they do for every method.
- **Cost.** Median 3.9 s per record, peak GPU memory 21 GB, and no out-of-memory errors across 3,200 records.

## 4. Review: these numbers probably understate TriAttention

Most of the scoring code follows the authors' code closely. The calibration does not:

- **Degenerate calibration text.** The Nov 10 statistics were computed on five sentences ("The quick brown fox…" plus four sentences about transformers) repeated until they reached 4,096 tokens. The authors' calibration guide (`docs/calibration.md` in WeianMao/triattention) explicitly says to *avoid degenerate inputs … the same sentence copied thousands of times*. It recommends natural text close to 32,768 tokens. The paper reports stable results from 50k to 960k tokens of real text. Query centers measured on a short loop of the same five sentences may not represent how Llama attends on real documents.
- **Combining heads.** For Llama's grouped heads (4 query heads per KV head), the paper z-scores each query head and keeps a key if any head ranks it highly (Eq. 12–13). The authors' code uses a "union" selection. This implementation averages the raw scores of the 4 query heads instead. That is a smaller difference, but it can drop keys that matter to only one head.
- **Setting mismatch (not a bug).** The paper prunes during generation, every 128 tokens, with a fixed token budget. In our protocol, as for every kvpress method, the document is compressed once. The paper's LongBench result (48.1 against 47.2 for the full cache on Qwen3-8B at 50%) is therefore not directly comparable with our 5–25% budgets on Llama. **There is no published TriAttention number for Llama-3.1-8B on LongBench.**

**Fix, ready to run.**

- `calibrate_triattention.py` now builds its calibration text from natural LongBench train-split documents, up to 32,768 tokens. No validation or test document is used, and nothing is repeated. It also runs the model body only, so it doesn't build a 32K × vocabulary output tensor.
- The harness takes `--triattention-stats`.
- `members/anees/run_triattention_recal.sh` recalibrates and reruns all four ratios, about 4.5 hours on the RTX 5090. Results go to `results/triattention/longbench_trainsplit_calibration/`, and the comparison labels them "recalibrated".
- Only the calibration changes. Head combining is kept as is, so any difference is due to the calibration alone.
- The Nov 10 results stay in the repo, and both sets will be reported.

Until the re-run is done, the measured flaws in `members/samruddhi/FLAWS.md` (large drops even at 25%, collapse on retrieval) should be read as flaws of *this calibration*, not proven flaws of the method.

## 5. Limitations

- Calibration as described in §4. The re-run is pending.
- Head combining differs from the paper (mean instead of z-score and max).
- One compression pass after reading, not the paper's pruning during generation.
- Memory is computed: kvpress masks evicted pairs instead of freeing them.
- The Nov 10 runs used WSL2 with FlashAttention, while the other kvpress runs used the lab's Docker image. Prompts, records and scorer are identical, but the software stacks differ slightly.
- Validation set only.

## 6. Plan for DATA 298B

1. Report the recalibrated results next to the Nov 10 ones.
2. If TriAttention still trails KVzip by a wide margin, implement the paper's z-score and max head combining and test it as a separate change.
3. Try pruning during generation (the paper's setting) on a long-generation task, where the method is designed to work.
4. Run the best setting on the held-out test split.

## Reproducing

```bash
python members/samruddhi/calibrate_triattention.py --output members/samruddhi/calibration/llama31_8b_stats_trainsplit.pt
python harness/run_longbench.py --press triattention --compression-ratio 0.75 \
    --triattention-stats members/samruddhi/calibration/llama31_8b_stats_trainsplit.pt \
    --output-dir results/triattention/longbench_trainsplit_calibration          # also 0.8, 0.9, 0.95
# or all of the above, restartable:
nohup bash members/anees/run_triattention_recal.sh > triattention_recal.log 2>&1 &
```

To reproduce the Nov 10 statistics: `calibrate_triattention.py --repeated-sample --max-length 4096`.
