# Team harness integration status

This is Vedant's TurboQuant reference package, version 0.2.0. TurboQuant is registered
in the shared harness as `turboquant_{2,3,4}bit` and `turboquant_mse_{3,4}bit`
(`turboquant_press.py`, `harness/run_longbench.py`). Use those runs for the team table.

## Shared harness path vs standalone runner

| Setting | Shared harness (`turboquant_*` presses) | Standalone `turboquant-longbench` |
|---|---|---|
| Dataset | Team 800-record dev file | Same file (SHA-256 checked) |
| Prompt construction | Document as context, question split off, answer cue for no-template tasks | Single formatted prompt |
| Precision | BF16 | BF16 by default (was FP16) |
| What is quantized | Document keys and values only | Every cached token, including question and generated tokens |
| Prefill attention | Full precision; stored cache quantized | Full precision for the tokens of each forward pass; stored cache quantized |
| Scoring | `harness/longbench.py`, 0–1, rapidfuzz | Pinned upstream metrics, 0–100, fuzzywuzzy |
| Memory | `kv_cache_gb`: packed document bytes + bf16 question | Packed bytes of the whole cache + quantizer state |

The two paths use the same `TurboQuantCacheLayer`, seeds and fp16 norms, so the document is
quantized identically. Do not mix their scores in one table: the scorers differ in scale
and in the code-similarity backend.

## Baseline

The team's full-cache baseline on all 800 records is 47.72
(`results/longbench/Llama-3.1-8B-Instruct_none_0.0.summary.json` on the team repo's main).
This package's scorer gives 47.45 on the same predictions; only lcc and repobench-p differ.

## Known gaps

- The paper's 2.5- and 3.5-bit settings split out outlier channels and give them more bits.
  This package uses one bit width per layer, so its 2/3/4-bit results are not those settings.
- The cache keeps decoded keys and values for attention, so packed storage savings are
  computed, not measured, as for KVzip and RestoreKV in the harness.

## Running the standalone tools

Use a virtual environment in this directory that reuses the lab image's torch:

```bash
cd members/vedant
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[dev,hf,plots,score]'
python -m pytest -q
```

Label standalone outputs as reference experiments, not shared-harness results.
