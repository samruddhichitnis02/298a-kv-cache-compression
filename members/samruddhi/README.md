# TriAttention on LongBench

Testing the **TriAttention** KV cache compression method on **LongBench** with
**Llama-3.1-8B-Instruct**. This page tells the whole story in plain language: what the method does,
what results it gives, the GPU memory crash we hit, why it happened, and how we fixed it.

| | |
|---|---|
| **Method** | TriAttention ([arXiv:2604.04921](https://arxiv.org/abs/2604.04921), [repo](https://github.com/WeianMao/triattention)) |
| **Model** | Llama-3.1-8B-Instruct (bf16) |
| **Benchmark** | LongBench v1 — 16 tasks, 800 documents |
| **GPU** | NVIDIA RTX 5090 (32 GB) |
| **Final run** | all 800 docs × 4 compression ratios, **0 crashes** |

---

## 📌 The idea in one minute

When a model reads a document, it stores two numbers (a **key** and a **value**) for every word it
sees. This memory is the **KV cache**, and it grows with document length until it dominates GPU memory.

**Compression** shrinks that cache. The trade is always the same:

> **Save memory → but how much quality do you lose?**

TriAttention shrinks the cache by **keeping only the most useful words and dropping the rest**. How
aggressively it drops is set by a compression ratio:

| Ratio | Keeps | Drops |
|:---:|:---:|:---:|
| `0.75` | 25% | 75% |
| `0.80` | 20% | 80% |
| `0.90` | 10% | 90% |
| `0.95` | 5% | 95% |

---

## 🏆 The results

All 800 documents, all four ratios, measured against the full (uncompressed) cache.

![Accuracy vs KV cache memory](results_out/figures/score_vs_kv_cache.png)

| Setting | Quality /100 | KV cache | Accuracy kept | Memory saved |
|:---|:---:|:---:|:---:|:---:|
| **Full cache** | **47.7** | 1.357 GB | 100% | — |
| TriAttention @0.75 | 38.7 | 0.355 GB | 81% | 74% |
| TriAttention @0.80 | 37.8 | 0.288 GB | 79% | 79% |
| TriAttention @0.90 | 33.1 | 0.154 GB | 69% | 89% |
| TriAttention @0.95 | 31.4 | 0.087 GB | 66% | 94% |

**Headline:** TriAttention shrinks the KV cache up to about **15×** (1.357 GB → 0.087 GB) while keeping
roughly **two‑thirds** of the quality, and much more at gentler settings.

Watch the trade play out as compression increases:

![Accuracy vs memory animation](results_out/figures/accuracy_vs_memory.gif)

Memory saved versus accuracy kept, side by side:

![Memory saved vs accuracy kept](results_out/figures/tradeoff.png)

Macro average score for each run:

![Macro average per run](results_out/figures/macro_average.png)

---

## 🎯 Where it shines, where it struggles

At the most aggressive setting (keep only 5%), here is how each task changed versus the full cache.
**Green = held up. Red = got worse.**

![Per-task impact](results_out/figures/per_task_delta.png)

**Holds up well** 👍

| Task | Full cache | @0.95 | Change |
|:---|:---:|:---:|:---:|
| triviaqa | 93.2 | 92.2 | ≈ none |
| samsum | 49.3 | 44.3 | small |
| repobench‑p (code) | 49.3 | 48.0 | small |

**Struggles** 👎

| Task | Full cache | @0.95 | Change |
|:---|:---:|:---:|:---:|
| passage_retrieval_en | 100.0 | 36.0 | big drop |
| qasper (QA) | 41.6 | 13.3 | big drop |
| multifieldqa_en (QA) | 54.9 | 21.8 | big drop |

**Why question‑answering struggles:** a QA answer usually depends on a few exact sentences hidden in a
long document. TriAttention keeps words by a general "looks important" score, not by the specific
question, so it can drop the one sentence that held the answer. Tasks whose meaning is spread across
many words (summaries, code) barely notice a few missing words.

Full per‑task scores at every ratio:

![Per-task scores](results_out/figures/per_task_scores.png)

The same scores animated, one frame per ratio:

![Per-task sweep animation](results_out/figures/per_task_sweep.gif)

---

## 🧭 The journey (and the big problem we solved)

```mermaid
flowchart TD
    A["Experiment 1<br/>short docs, 260 records<br/>✅ worked"] --> B["Try the full 800 docs<br/>some up to 32K tokens"]
    B --> C["❌ GPU out of memory<br/>on the long documents"]
    C --> D["Guess 1: scoring too big?"]
    C --> E["Guess 2: GPU too small?"]
    D --> F["Profile the memory properly"]
    E --> F
    F --> G["Real cause:<br/>Windows PyTorch had no<br/>efficient attention kernel<br/>→ tried to grab 122 GB"]
    G --> H["Fix: run in WSL2 Linux<br/>which HAS FlashAttention"]
    H --> I["✅ All 800 × 4 ratios<br/>0 crashes, ~21 GB peak"]
```

### Step 1 — Short documents worked

We first ran only the shorter LongBench documents (260 of them). They ran fine and gave the first
accuracy‑vs‑memory picture. We also replaced an early shortcut (guessing TriAttention's statistics)
with the authors' real calibration, so every number reflects the method as designed.

### Step 2 — The full 800 kept crashing

Adding the long documents (up to **32,000 tokens**) caused constant **out‑of‑memory crashes**. On the
hardest setting about **half** the documents failed. Failed documents score zero, which wrecked the
averages. We could not get a clean result.

### Step 3 — Two wrong guesses

> **Guess 1 — "TriAttention's math is too big."** We rewrote its scoring to work in small chunks (and
> proved the result was identical, difference = 0.000). It helped, but the long docs still crashed.

> **Guess 2 — "The 32 GB GPU is too small."** The math said no. Keeping 25% of a 32K document is only
> about **1 GB** of cache; with the 16 GB model that is ~17 GB, nowhere near 32 GB. A 38 GB crash made
> no sense. So we stopped guessing and measured.

### Step 4 — The real cause

Profiling one 32K document showed that **even with no compression at all**, the model tried to grab
**122 GB** at the very first step.

Here is why. To read a document the model compares every word with every other word. For 32,000 words
that is a 32,000 × 32,000 table = **122 GB**. Modern GPUs avoid building that table using a shortcut
called **FlashAttention**, which does the same thing in about **21 GB**.

The catch: **our Windows PyTorch did not have FlashAttention** ("Torch was not compiled with flash
attention"). With no shortcut available, it fell back to the giant 122 GB table and crashed.

![Why it crashed on Windows and why WSL fixed it](results_out/figures/memory_story.png)

This also explained a mystery: a teammate's runs on the **lab Linux machine** read the same 32K
documents at ~24 GB with no crashes. Same GPU, same code. The only difference was that Linux PyTorch
**had** FlashAttention and Windows did not.

> **Conclusion:** the crash was not a bug in TriAttention and not a too‑small GPU. It was a **missing
> attention kernel in the Windows build**.

### Step 5 — The fix: WSL2 + FlashAttention

Rather than buy a bigger GPU (which would not help, since the 122 GB table overflows any card), we ran
inside **WSL2** (Linux inside Windows, same RTX 5090) with a PyTorch build that **has FlashAttention**.

The worst‑case 32K document then ran in **21 GB, single shot, no crash**. We ran the full benchmark:

> **All 800 documents × 4 ratios = 3,200 runs. Zero crashes. Peak ~21–24 GB.**

Those are the numbers at the top of this page.

---

## 🧪 What each attempt actually did

| Attempt | Idea | Fixed the crash? |
|:---|:---|:---:|
| Faithful calibration | use the authors' real statistics | fixed the *numbers*, not the crash |
| Chunked scoring | compute the score in small pieces | helped, proven identical, not enough alone |
| Chunked prefill | feed the document in pieces | helped hard ratios, not a fair comparison |
| CPU offloading | keep the cache in normal RAM | worked but slow and hard to compare |
| **WSL2 + FlashAttention** | **use the efficient attention kernel** | ✅ **this fixed it** |

---

## 📂 What is in this folder

| Path | What it is |
|:---|:---|
| `triattention_press.py` | the method: scores words and keeps the best |
| `calibrate_triattention.py`, `calibration/` | computes and stores the method's statistics |
| `build_dev_data.py` | prepares the 800 LongBench documents |
| `run_all.py` | runs every ratio, then the analysis |
| `analyze_results.py` | builds the CSVs, charts and GIFs |
| `write_flaws.py` | writes `FLAWS.md` from the measured numbers |
| `results_out/` | all outputs: CSVs, charts, animations, log |
| `RESULTS.md` / `FLAWS.md` | result tables, and the measured weaknesses |
| `experiments_archive/` | everything from the earlier experiments and the OOM investigation |

Result files for the four ratios live at the repo level in `results/triattention/longbench/`
(mirroring `results/restorekv/`).

---

## ▶️ How to reproduce (WSL2 + FlashAttention)

```bash
cd /mnt/c/Users/samm/Documents/Samruddhi/298a-kv-cache-compression
HF_TOKEN=<token> ~/flashattn-env/bin/python members/samruddhi/build_dev_data.py
HF_TOKEN=<token> ~/flashattn-env/bin/python members/samruddhi/calibrate_triattention.py \
    --output members/samruddhi/calibration/llama31_8b_stats.pt
HF_TOKEN=<token> ~/flashattn-env/bin/python members/samruddhi/run_all.py \
    --compression-ratios 0.75,0.8,0.9,0.95 --skip-baseline
```

No `--max-prompt-tokens` (all 800 docs) and no `--chunk-prefill` (normal single‑shot, same as the
baseline, so the comparison is fair).

---

## ✅ The takeaway

TriAttention trades memory for accuracy well (up to **~15× smaller** cache while keeping **~66%** of
quality, more at gentler settings). Its weak spot is **pinpoint question‑answering**. And the scary
crash was never about the method or the GPU size: it was a **missing Windows attention kernel**, fixed
cleanly by moving to **WSL2 with FlashAttention**.
