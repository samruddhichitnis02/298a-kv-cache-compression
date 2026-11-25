# Shared tools for the four review points

Five tools, built once, so every method is measured the same way. Each member runs them on their own method.

| Review point | Tool | Where |
|---|---|---|
| 1. Hybrid eviction + quantization at actual bytes | Packed cache | `--pack-bits`, `--memory-budget` on `run_longbench.py` and `run_needles.py` |
| 2. Task- and budget-adaptive allocation | Adaptive rule | `harness/adaptive.py`, `--adaptive-policy` |
| 3. Serving measurements | Timing | `--timing` on `run_longbench.py` |
| 4. Many questions per cache | Many-questions runner | `harness/run_needles.py --tasks many` |
| 4. Longer documents | Needle documents at any length, chunked prefill | `harness/needles.py`, `--chunk-prefill` |

All five run through one driver, `harness/serve.py`: prefill the document under the press, optionally pack the cache, answer one or more questions, and time each step.

**Status.** Unit tests pass (38: 25 in `harness/`, 13 in `members/manjot/test_rdkv.py`). On November 25 the smoke run of all four points finished on Llama-3.1-8B on the lab RTX 5090 with no failed run (29 runs, commit `8c257cd`), and 64,000-token documents fit with `--chunk-prefill 8192` (peak 28.66 GB with the full cache). Smoke runs are one or two records each, so no full-size run has finished yet. Still do a short smoke run before a long one.

## One command per point

`harness/sprint/run_point.sh` runs a point across the methods in the shared harness:

```bash
bash harness/sprint/run_point.sh timing smoke     # a few short records first
nohup bash harness/sprint/run_point.sh timing > timing.log 2>&1 &
```

Points: `hybrid` (point 1), `timing` (point 3), `many` and `long` (point 4). The sections below explain what each one runs and the flags behind it.

## Before you start

```bash
pip install -r requirements.txt     # the lab image has torch only; this keeps its torch build
python -m pytest -q harness/test_serve.py harness/test_tools.py members/manjot/test_rdkv.py     # 38 tests, CPU
```

The lab image has no `unzip`: unpack the data with `python -m zipfile -e ../processed_longbench.zip data/`.

Shared record lists (`harness/subsets/`):

- `timing_100.json`: 100 validation records for timing, from 180 to 32,000 tokens.
- `train_300.json`: 300 records outside the validation and test splits, for fitting the adaptive rule. They are in the full file, so pass `--data data/processed_longbench/processed_longbench.jsonl`.

New modes write to their own files (`..._pack4.jsonl`, `..._timing.jsonl`, `..._adaptive0.9.jsonl`), so they never mix with the existing results. Use `--output-dir results/<method>/<experiment>` as usual. Every summary now records the GPU name and the code commit.

## Point 1: hybrid at measured bytes

Today, KVzip, RestoreKV and RDKV only hide evicted pairs, so GPU memory does not drop and sizes are computed. `--pack-bits` makes the cache real: evicted pairs are removed, the rest are stored at 16 bits (lossless) or packed at 8, 4 or 2 bits, and the original tensors are freed. `--memory-budget` sets the eviction ratio per document so the packed cache hits a byte budget.

```bash
# the three strategies at 10% of the full document cache, in measured bytes
python harness/run_longbench.py --press kvzip --pack-bits 16 --memory-budget 0.1     # eviction only
python harness/run_longbench.py --press kvzip --pack-bits 2  --memory-budget 0.1     # hybrid: more tokens, 2 bits
python harness/run_longbench.py --press kvzip --pack-bits 4  --memory-budget 0.1     # hybrid: 4 bits
python harness/run_longbench.py --press none  --pack-bits 2                          # quantization only (about 19%)
```

Each row gets `document_cache_share_measured` (packed document bytes over the full document cache), `kv_cache_gb_measured` (document plus question), `eviction_ratio` and `kept_pairs_share`. For packed runs `kv_cache_gb` is the measured value.

- **Quantization only cannot reach every budget.** With scales and zero points, 4-bit is about 31% of the full cache and 2-bit about 19%. Report those as they are.
- **Which presses work.** Any press that leaves a normal cache: `none`, `snapkv`, `kvzip`, `restorekv`, `triattention`. `restorekv_4bit`, `restorekv_2bit` and TurboQuant already quantize, so they only take `--pack-bits 16`. ResidualKV runs outside this harness.
- **RDKV has its own packing,** because it mixes bit-widths: `--press rdkv --compression-ratio 0.9 --rdkv-packed` writes each layer into real packed buffers (`members/manjot/rdkv_pack.py`), attends to what they decode to, and reports their size in the same `document_cache_share_measured` column. `--rdkv-bitset 0,16` gives RDKV's eviction-only half. On a small model the measured size was 10.6% where RDKV's own formula says 10.5%.
- **TurboQuant** reports its packed codes as the measured size. Its cache also keeps a decoded copy for speed, recorded as `decoded_copy_gb`.
- **Lossless check.** With `--pack-bits 16` the answers should match the masked run. On the small model they matched on 5 of 7 records with KVzip (two long code completions diverged late) and 7 of 7 with SnapKV. Hiding a pair with a fake key is not exactly the same arithmetic as removing it.
- **Cost.** Each decoding step unpacks every layer. That time is in `tokens_per_s` when you add `--timing`.

## Point 2: adaptive budgets

The rule groups prompts into content types from the prompt alone (length and text statistics, plus an optional signal from your method) and gives each type its own setting, with the average memory equal to a fixed setting's.

```bash
# no GPU: evaluate on runs you already have at several ratios
python harness/adaptive.py evaluate --target 0.9 --runs \
    results/rdkv/longbench/Llama-3.1-8B-Instruct_rdkv_{0.75,0.8,0.9,0.95}.jsonl

# live: fit on the training sample, then run with the policy
python harness/run_longbench.py --press rdkv --compression-ratio 0.9 --records harness/subsets/train_300.json \
    --data data/processed_longbench/processed_longbench.jsonl --output-dir results/rdkv/train_sample     # x4 ratios
python harness/adaptive.py fit --target 0.9 --runs results/rdkv/train_sample/*.jsonl \
    --data data/processed_longbench/processed_longbench.jsonl --out results/rdkv/adaptive_policy.json
python harness/run_longbench.py --press rdkv --adaptive-policy results/rdkv/adaptive_policy.json
```

`evaluate` is exact, not an estimate: decoding is greedy, so the adaptive score is a lookup of each record's existing result at its chosen ratio. It splits each task in half, fits on one half and scores the other.

What it gives on the committed validation results (mean over 20 random splits, `--repeats 20`; the ± is the spread of the gain):

| Method | Memory | Fixed | Adaptive, no task label | Per task, held out |
|---|---|---|---|---|
| RestoreKV | 11.4% | 43.29 | 45.32 (+2.02 ± 0.58) | 45.32 |
| KVzip | 11.4% | 41.61 | 44.32 (+2.71 ± 0.66) | 44.50 |
| RDKV | 11.6% | 42.58 | 44.07 (+1.49 ± 0.47) | 44.25 |
| SnapKV | 11.4% | 32.64 | 34.86 (+2.23 ± 0.45) | 34.90 |
| TriAttention | 11.4% | 33.10 | 36.28 (+3.18 ± 0.37) | 36.67 |

At about 11% memory the rule gains 1.5 to 3.2 points without the task label, as much as knowing the task. Where a fixed setting is already within about a point of the full cache (4-bit RestoreKV, TurboQuant, ResidualKV, and every method at about 21% memory) it gains nothing. The full table is in `results/comparison/TASK_ADAPTIVE.md`.

Result files that mix presses or bit-widths work too: settings are then named `<press>@<ratio>`, for example `--target restorekv_4bit@0.95`, and the rule chooses the strategy as well as the budget. The number of content types (`--clusters`, default 24) was chosen on these validation results. `--signal` is the place to add your method's own per-record number: a JSON file mapping `record_id` to a number or list.

## Point 3: serving measurements

```bash
python harness/run_longbench.py --press snapkv --compression-ratio 0.9 --timing --records harness/subsets/timing_100.json
```

Each row gets a `timing` block, the median over `--timing-repeats` runs (default 3) after one unrecorded warm-up:

| Field | Meaning |
|---|---|
| `prefill_s` | Document forward pass plus the press's own work (scoring, eviction) |
| `pack_s` | Packing time, with `--pack-bits` |
| `first_token_s` | From the question to the first generated token |
| `ttft_s` | `prefill_s + pack_s + first_token_s` |
| `tokens_per_s` | Decoding speed after the first token, including any unpacking |
| `document_cache_gb_measured` | Bytes of the tensors the cache really holds |
| `peak_gpu_gb`, `model_gpu_gb` | Peak GPU memory for the record, and the model alone |

The last two rows of the table, `document_cache_gb_measured`, `peak_gpu_gb` and `model_gpu_gb`, are fields of the row itself, next to `timing`, not inside it.

- **Compression overhead** is your method's `prefill_s` minus the full cache's `prefill_s` on the same record. Run `--press none --timing` once on the same machine.
- **Measured memory without packing** is the full size for presses that only hide pairs. That is the honest number: say "simulated" for those, or pack them.
- Run all timing on one machine. `seconds` in a timing row covers all repeats.

## Point 4: many questions and long documents

`harness/needles.py` builds documents of any length from LongBench prose outside the validation and test splits, with facts inserted at spread-out depths. This is RULER-style, generated by us: the public RULER build we use stops at 16K and asks one question per document.

```bash
python harness/needles.py --length 32768          # once per length -> data/needles/needles_32768.jsonl
python harness/run_needles.py --length 32768 --press kvzip --compression-ratio 0.9
python harness/run_needles.py --length 65536 --press none --chunk-prefill 8192
python harness/run_needles.py --length 32768 --press snapkv --compression-ratio 0.9 --tasks many --compress-with-first-question
```

Tasks: `single`, `multikey`, `multivalue`, `multiquery` (one question each) and `many` (16 questions on one document). Twenty documents per task by default. The same command gives the same file; check the printed sha256 against a teammate's.

- **Many questions.** The document is compressed once, before any question, and all 16 are answered from that cache. The summary gives the score over the first 1, 2, 4, 8 and 16 questions, the worst question, and the seconds per question when the compression cost is shared.
- **Question-aware contrast.** `--compress-with-first-question` appends the first question before compressing, which is how a question-aware method would be used. Later questions then show what was lost.
- **Long documents.** `--chunk-prefill N` feeds the document in pieces so it fits in memory. It gives the same answers as one pass (checked on the small model, with and without KVzip). It works for `none`, `kvzip` and `restorekv`. SnapKV, TriAttention and RDKV compress inside the prefill pass and need the whole document at once, so their limit is whatever fits on the GPU.
- `--pack-bits` and `--memory-budget` work here too.

## Tables for the report

```bash
python harness/sprint/make_tables.py            # every point that has results
python harness/sprint/make_tables.py timing
```

Writes one Markdown table per point to `results/sprint/`, straight from the result files: score against measured bytes (point 1), timings with overhead against the full cache and sessions per 32 GB (point 3), score by number of questions (point 4), and score by document length (point 4).

## ResidualKV

ResidualKV runs on the authors' code, so it has its own runner for the needle documents: `members/charvee/run_residualkv_needles.py`. It compresses the document alone, then answers each question from a copy of that cache, and writes the same row format. It has not been run on a GPU yet; its docstring lists what to check first.

## Known limits

- Not yet run on a GPU or on Llama-3.1-8B. Timing uses a GPU synchronize call that the laptop tests never exercise on CUDA.
- Whether a 64K document fits on a 32 GB card, even with chunked prefill, is untested. The full cache alone is about 8 GB at 64K and 16 GB at 128K, on top of the 16 GB model.
- ResidualKV's real packed size is still its owner's to measure; its runner reports the size of the tensors its cache holds.
- The adaptive rule's number of content types was chosen on validation results; fitting on the training sample is the cleaner test and needs GPU runs.
