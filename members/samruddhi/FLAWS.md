# TriAttention: Observed Flaws

Every flaw below is measured from our own full-800 LongBench runs on Llama-3.1-8B-Instruct, compared
against the full (uncompressed) cache and against RestoreKV at the same budget. Numbers come from
`results/triattention/longbench/`, `results/longbench/`, and `results/restorekv/longbench/`.

Baseline for reference: full cache macro average = **47.7**.
TriAttention macro average: **38.7 (keep 25%) → 37.8 → 33.1 → 31.4 (keep 5%)**.

---

## Flaw 1: Large accuracy drop even at light compression

TriAttention loses a lot of quality sooner than it should. Keeping 25% of the cache already drops the
score from 47.7 to 38.7, a 9-point fall. By comparison RestoreKV at the same 25% budget scores 47.3,
almost no loss. So TriAttention pays a steep price the moment it starts compressing.

| Keep | TriAttention | RestoreKV | Full cache |
|:---:|:---:|:---:|:---:|
| 25% | 38.7 | 47.3 | 47.7 |
| 5% | 31.4 | 36.1 | 47.7 |

---

## Flaw 2: Question-answering collapses

The tasks that need a few exact sentences from the document fall the hardest. At 95% eviction:

| Task | Full cache | TriAttention | Drop |
|:---|:---:|:---:|:---:|
| qasper | 41.6 | 13.3 | −28.3 |
| multifieldqa_en | 54.9 | 21.8 | −33.0 |
| 2wikimqa | 45.5 | 19.7 | −25.8 |
| trec | 70.0 | 36.0 | −34.0 |

**Why:** TriAttention keeps words using a general importance score learned from the model's attention
geometry. That score does not know the specific question being asked, so the one sentence that holds
the answer can look unimportant and get thrown away. RestoreKV keeps what the recent query actually
attended to, so it holds the answer sentence better (qasper 40.6 vs 13.3, a 27-point gap).

---

## Flaw 3: Worst case is a near-total collapse on positional retrieval

On `passage_retrieval_en`, which asks the model to find a passage by its position, TriAttention drops
from **100.0 to 36.0** at 95% eviction, the single biggest fall of any task. Even though TriAttention
beats RestoreKV here (RestoreKV drops to 14.0), losing 64 points shows the method cannot preserve the
exact positional evidence this task depends on once most of the cache is gone.

---

## Flaw 4: The score is unstable, not smoothly monotonic

A well-behaved method should score at least as well when it keeps more cache. TriAttention does not
always do this. On `hotpotqa` it scores **35.6 while keeping 25%** but **41.3 while keeping only 5%** —
keeping *more* cache gave a *worse* answer. This non-monotonic behavior means the importance score is
noisy: which tokens get kept does not line up cleanly with which tokens actually matter, so adding more
tokens can add distracting ones.

---

## Flaw 5: It needs an offline calibration step

Unlike eviction methods that work out of the box, TriAttention first needs a one-time calibration pass
to compute per-head frequency statistics for the model before it can score anything. If those
statistics are wrong or missing, the method behaves badly (an early guessed-statistics version of ours
produced clearly wrong scores). This is an extra dependency and an extra place for things to go wrong.

---

## Flaw 6 (environment, not the algorithm): heavy prefill on long inputs

TriAttention evicts tokens only after the model has read the whole document, so on very long inputs the
full read still happens first. On our 32 GB GPU this needed an efficient attention kernel
(FlashAttention) to fit a 32K-token document; without it the read tried to allocate 122 GB and crashed.
This is a property of reading long inputs in general, not unique to TriAttention, but it means the
method gives no memory relief during the initial read of a long prompt.

---

## Summary

| Flaw | Evidence |
|:---|:---|
| Big accuracy drop even at light compression | 47.7 → 38.7 at just 25% kept |
| Question-answering collapses | qasper 41.6 → 13.3; multifieldqa 54.9 → 21.8 |
| Near-total collapse on positional retrieval | passage_retrieval 100 → 36 |
| Unstable, non-monotonic score | hotpotqa worse with more cache (35.6 vs 41.3) |
| Needs offline calibration | wrong stats produce wrong scores |
| No memory relief during the long-prompt read | 32K read needs FlashAttention or it OOMs |

**The core weakness:** TriAttention's importance score is not tied to the actual question, so it is
good at keeping broadly useful information (retrieval, code, dialogue) but bad at keeping the one exact
fact a specific question needs, and its token choices are noisy enough to sometimes get worse with more
budget.
