# Experiments Archive - TriAttention on LongBench

Everything from the earlier experiments and the OOM/memory investigation is preserved here:
all code, results, figures, logs, and the MD writeups. The active `members/samruddhi/` folder was
cleared so the final run (WSL2 + FlashAttention, single-shot, full 800 records) starts clean. Fresh
code is restored from git for that run.

## The journey, in order

1. **260-record subset run (native Windows).** TriAttention on LongBench prompts under 6K tokens,
   vs the full-cache baseline. Produced the accuracy/memory tradeoff and the flaw analysis
   (QA degrades most). See `RESULTS.md`, `FLAWS.md`, `results_out/`.

2. **Faithful calibration.** Replaced an early online-estimate of the query statistics with the
   authors' real calibration (per-head pre-RoPE Q/K frequency stats), correcting
   invented/exaggerated flaws.

3. **Full-800 attempt + OOM investigation.** Extending to all 800 records (prompts up to 32K tokens)
   hit GPU OOM on native Windows. Root cause: NOT TriAttention and NOT the kept cache, but the
   native-Windows PyTorch build lacking a usable memory-efficient attention kernel (FlashAttention
   not compiled in; efficient SDPA rejected by the model's call path) -> the math attention path
   tried a 122 GB allocation for a 32K prefill.

4. **Memory workarounds tried (native Windows).** Chunked prefill, chunked scoring (verified
   numerically identical, max diff 0.000), CPU KV-cache offloading for the baseline, score-tensor
   shrink. Aggressive ratios (0.9/0.95) ran; light ratios (0.75/0.8) stayed OOM-prone - a software
   environment limit.

5. **Resolution: WSL2 + FlashAttention.** In WSL2 Ubuntu with a FlashAttention-capable torch build,
   the worst-case 32K record runs single-shot at ~21 GB (no chunking, no OOM). That is the setup
   used for the final apples-to-apples full-800 run.

## Contents

**Code**
- `triattention_press.py` - TriAttention as a kvpress ScorerPress (calibrated trig scoring; chunked scorer)
- `calibrate_triattention.py` + `calibration/` - the authors' calibration and the produced stats
- `build_dev_data.py` - builds the LongBench dev set
- `run_all.py` - runs the baseline + TriAttention sweep + analysis
- `analyze_results.py`, `write_flaws.py` - CSVs/figures/RESULTS.md, and FLAWS.md
- `chunked_prefill.py` - native-Windows chunked-prefill workaround
- `run_offloaded_baseline.py` - CPU KV-cache offloaded full-cache baseline
- `make_pdf.py`, `selftest_triattention.py` - PDF summary + plumbing check

**Results / docs**
- `README.md` - full writeup incl. the OOM investigation
- `RESULTS.md`, `FLAWS.md` - subset results and measured flaws
- `results_out/` - per-record/per-task CSVs, figures, animations, run log
