# Task- and budget-adaptive allocation

Draft section for the final report. It answers the question raised in review: do summarization, retrieval, QA and code need different cache strategies? All numbers come from `results/comparison/TASK_ADAPTIVE.md`, which `results/comparison/task_adaptive.py` regenerates from committed result files. No new runs were needed.

## Setup

- **Data.** The 800-record LongBench validation set: 16 tasks in six categories, 50 records per task.
- **Methods.** RestoreKV (eviction only, and with 4-bit or 2-bit quantization), KVzip, RDKV, SnapKV and TriAttention, each at compression ratios 0.75, 0.8, 0.9 and 0.95. The quantized RestoreKV variants were run at 0.9 and 0.95 only.
- **Memory.** Computed cache size as a share of the full cache: about 26%, 21%, 11% and 6% at the four ratios.
- **Two questions.** Which method is best for each task type at a given memory? And how much is gained if each task gets its own budget?

## Finding 1: the best strategy depends on the task type

Points lost against the full cache at about 11% memory (negative is worse):

| Method | Single-doc QA | Multi-doc QA | Summarization | Few-shot | Synthetic | Code |
|---|---|---|---|---|---|---|
| RestoreKV 4-bit | -0.6 | -1.0 | +0.3 | -0.9 | -2.2 | +2.5 |
| RestoreKV (eviction only) | +0.3 | -4.8 | -0.4 | -4.0 | -22.2 | +0.1 |
| RDKV | -10.2 | -5.7 | -3.3 | -0.9 | -11.7 | +0.8 |
| KVzip | -4.1 | -2.2 | -1.4 | -8.9 | -23.8 | -0.1 |
| SnapKV | -22.4 | -18.6 | -7.5 | -8.3 | -33.2 | -2.2 |

- **Summarization and code tolerate heavy compression.** Every method except SnapKV and TriAttention loses at most 3.3 points on either at 11% memory, and RestoreKV with quantization loses less than 1 point on either even at 6%.
- **Synthetic retrieval breaks under eviction.** Eviction-only RestoreKV loses 22 points at 11% memory and 43 points at 6%. Keeping more tokens at fewer bits repairs it: the 4-bit variant loses 2.2 points at 11%, and at 6% the 2-bit variant is the best method (-2.0).
- **Few-shot tasks favour RDKV.** It is best or tied for best in this category at all four ratios, losing 0.9 points at 11% and 3.2 at 6%.
- **Question answering needs the hybrid.** At 6% memory the 4-bit variant is the only method within 2 points of the full cache on multi-document QA.

Each category holds two or three tasks (100 to 150 records), so differences of one or two points are within noise. The large gaps above are not.

## Finding 2: per-task budgets help most at small memory

Each task was allowed its own ratio of the same method, with total memory held at or below that of one fixed ratio. "Held out" chooses the ratios on half of each task's records and scores on the other half. "Upper bound" chooses and scores on the same records.

| Method | Memory | Fixed ratio | Per-task, held out | Per-task, upper bound |
|---|---|---|---|---|
| RestoreKV | 11.4% | 43.29 | 45.78 (+2.49) | 46.89 (+3.59) |
| KVzip | 11.4% | 41.61 | 45.03 (+3.42) | 45.88 (+4.27) |
| RDKV | 11.6% | 42.58 | 43.82 (+1.24) | 45.66 (+3.08) |
| SnapKV | 11.4% | 32.64 | 34.74 (+2.10) | 36.47 (+3.83) |
| TriAttention | 11.4% | 33.10 | 36.70 (+3.60) | 37.83 (+4.73) |

- **At about 11% memory every method gains**, by 1.2 to 3.6 points on held-out records.
- **At about 21% memory the gain disappears.** Held-out changes range from -0.5 to +1.0 points, because every method is already close to the full cache there.
- **The allocation is consistent across methods.** Passage retrieval gets the most or second-most memory under every method. NarrativeQA, QMSum and TriviaQA get the least under every method.

## Limits

- The allocation uses the task label, which a deployed system does not have. These numbers bound what a rule that reads only the prompt can reach; that rule is the next step.
- Memory is computed, not measured on the GPU.
- ResidualKV and TurboQuant are not included, because their memory does not line up with the four ratios.
- TriAttention uses the Nov 10 calibration, which likely understates the method.
