# Linear issues to paste (Charvee — ResidualKV)

Copy each block into Linear (or give this file to your Linear agent).

---

## Issue 1 (parent) — ResidualKV LongBench evaluation

**Title:** ResidualKV: run LongBench 800-record eval vs full-cache baseline

**Description:**
```markdown
## Goal
Evaluate ResidualKV (formerly DeltaKV, arXiv 2602.08005) on Llama-3.1-8B-Instruct using the shared team LongBench protocol so scores compare to the full-cache baseline.

## Owner
Charvee Saraiya (`members/charvee/`)

## Method
- Authors' code: CURRENTF/DeltaKV (vendored under `members/charvee/DeltaKV/`, target commit `11a837b`)
- Compressor: `JitaiHao/Llama-3.1-8B-Instruct-Compressor`
- Runner: `members/charvee/run_residualkv.py`
- Backend: HF + `DELTAKV_HF_ATTN_IMPLEMENTATION=sdpa` (no flash-attn / Sparse-vLLM required for this eval)

## Evaluation protocol (must match team)
- Dataset: `data/processed_longbench/processed_longbench_dev.jsonl` (**800** records = 16 tasks × 50)
- Same prompts + chat-template rules + team harness scorer (`harness/longbench.py`)
- Do **not** use full original LongBench download or DeltaKV's buggy first-line scorer path for trec/triviaqa/samsum
- Baseline (do not rerun): `results/longbench/Llama-3.1-8B-Instruct_none_0.0.jsonl`
  - **n_records = 800**, **average = 47.72**

## Compression ratios
| ratio | approx keep |
|------:|------------:|
| 0.75 | 25% |
| 0.80 | 20% |
| 0.90 | 10% |
| 0.95 | 5% |

Priority if GPU time is short: **0.9 then 0.75**, then 0.8 and 0.95.

## Metrics to report (not accuracy/ROC)
- Per-task LongBench score
- Overall / category average
- Per-record `kv_cache_gb` (GB = 1e9 bytes) and `full_kv_cache_gb`
- Per-record `seconds`
- Δ vs baseline 47.72 at each ratio

## Outputs
`results/residualkv/longbench/Llama-3.1-8B-Instruct_residualkv_<ratio>.jsonl` (+ `.summary.json`)

## Acceptance
- [ ] Smoke (`--per-task 2`) succeeds on lab/cloud GPU
- [ ] Full 800-record runs for 0.9 and 0.75
- [ ] Full 800-record runs for 0.8 and 0.95 (if time)
- [ ] Results uploaded to Drive / committed under team results path
- [ ] Report section filled with scores + kv_cache_gb vs baseline
```

---

## Issue 2 — Lab/cloud GPU setup + smoke

**Title:** ResidualKV: GPU setup and smoke test (--per-task 2)

**Description:**
```markdown
## Parent
ResidualKV LongBench evaluation

## Why
Confirm install, HF auth, DeltaKV import, and scoring path before spending ~hours on full 800-record runs.

## Where
SJSU lab RTX 5090 Docker **or** cloud GPU with SSH (≥24GB VRAM preferred). Not Mac CPU/MPS.

## Steps
1. Mount/sync `298a-kv-cache-compression` (+ DeltaKV under `members/charvee/DeltaKV` or `/app/DeltaKV`)
2. `pip install transformers==5.2.0 rouge rapidfuzz loguru` (try 5.12.1 only if import fails)
3. `hf auth login` (Llama 3.1 license accepted) — token only in terminal
4. Smoke:
   ```bash
   cd /app/298a-kv-cache-compression
   python members/charvee/run_residualkv.py --compression-ratio 0.9 --per-task 2
   ```
5. Expect ~32 progress lines and files under `results/residualkv/longbench/`

## Estimate
~10 min inference + one-time ~16GB model download

## Done when
- Smoke finishes without hard fail
- Sample jsonl rows include `record_id`, `task`, `prediction`, `score`, `kv_cache_gb`, `seconds`
```

---

## Issue 3 — Full 800 @ 0.9 and 0.75

**Title:** ResidualKV: full LongBench 800 at compression ratios 0.9 and 0.75

**Description:**
```markdown
## Parent
ResidualKV LongBench evaluation

## Runs
```bash
python members/charvee/run_residualkv.py --compression-ratio 0.9
python members/charvee/run_residualkv.py --compression-ratio 0.75
```
(or nohup loop; resume-safe — already-done `record_id`s are skipped)

## Estimate
~1.5–3 hours **per** ratio on RTX 5090

## Done when
- Each run has 800 rows in the jsonl
- `.summary.json` has per-task scores + average
- Median/mean `kv_cache_gb` noted for the report
- Compared against baseline average **47.72**
```

---

## Issue 4 — Full 800 @ 0.8 and 0.95

**Title:** ResidualKV: full LongBench 800 at compression ratios 0.8 and 0.95

**Description:**
```markdown
## Parent
ResidualKV LongBench evaluation

## Runs
```bash
python members/charvee/run_residualkv.py --compression-ratio 0.8
python members/charvee/run_residualkv.py --compression-ratio 0.95
```

## Estimate
~1.5–3 hours per ratio

## Done when
- 800 rows + summary for each ratio
- Results packed (`residualkv_results.tgz`) and uploaded to Drive / shared with Anees
```

---

## Issue 5 — Report section 4.4 metrics

**Title:** ResidualKV: write evaluation methods + metrics for report (rubric 4.4)

**Description:**
```markdown
## Parent
ResidualKV LongBench evaluation

## Content for report § Model Evaluation Method
State that accuracy/ROC/MSRE do **not** apply. Evaluation is long-context KV-cache compression quality + memory:

1. **Protocol:** shared 800-record LongBench English dev set; same prompts/scorer as team
2. **Quality metrics:** per-task LongBench scores; overall average; Δ vs full-cache baseline 47.72
3. **Efficiency metrics:** per-record `kv_cache_gb`, `full_kv_cache_gb`, `seconds`
4. **Conditions:** ratios 0.75/0.8/0.9/0.95; HF backend; Llama-3.1-8B-Instruct + ResidualKV compressor

## Done when
- Table of scores per ratio vs baseline
- Note that ratio ≠ exact memory % (full layers + sink/recent + residuals remain)
- Rubric 4.4 text drafted and reviewed
```

---

## Issue 6 (note) — SAM-65 / SAM-69 closure notes

**Title:** Add closure notes for SAM-65 and SAM-69 (ResidualKV setup)

**Description:**
```markdown
## Context
SAM-65 and SAM-69 were closed from Charvee's account on Nov 4 with no notes. Add short completion notes.

## Suggested notes to paste

### SAM-65
ResidualKV/DeltaKV project kickoff completed: paper + CURRENTF/DeltaKV code path identified; Llama-3.1-8B-Instruct compressor checkpoint located (`JitaiHao/Llama-3.1-8B-Instruct-Compressor`); member folder/runner path established under `members/charvee/`.

### SAM-69
G1 checked. Setup path confirmed with team harness (800-record `processed_longbench_dev.jsonl`, shared baseline 47.72). Can remain closed. Follow-on work tracked under ResidualKV LongBench evaluation parent issue (smoke + full ratio runs).
```
