# Team 6 pipeline monitoring report

Built 2026-11-12T06:34:36+00:00 from 27 result files.

## Stage counts

| Stage | Rows |
|---|---|
| Raw | 3,750 |
| Cleaned / transformed | 3,731 |
| Train / calibration | 2,131 |
| Validation | 800 |
| Test | 800 |

## Runs

| Run | Records | OOM | Macro score | Mean KV cache (GB) |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct_kvzip_0.75 | 800 | 0 | 47.43 | 0.355 |
| Llama-3.1-8B-Instruct_kvzip_0.9 | 800 | 0 | 41.61 | 0.154 |
| Llama-3.1-8B-Instruct_none_0.0 | 800 | 0 | 47.72 | 1.357 |
| Llama-3.1-8B-Instruct_restorekv_0.75 | 800 | 0 | 47.31 | 0.355 |
| Llama-3.1-8B-Instruct_restorekv_0.8 | 800 | 0 | 47.63 | 0.288 |
| Llama-3.1-8B-Instruct_restorekv_0.9 | 800 | 0 | 43.29 | 0.154 |
| Llama-3.1-8B-Instruct_restorekv_0.95 | 800 | 0 | 36.13 | 0.087 |
| Llama-3.1-8B-Instruct_restorekv_4bit_0.95 | 800 | 0 | 46.63 | 0.087 |
| Llama-3.1-8B-Instruct_snapkv_0.75 | 800 | 0 | 39.40 | 0.355 |
| Llama-3.1-8B-Instruct_snapkv_0.9 | 800 | 0 | 32.64 | 0.154 |
| Llama-3.1-8B-Instruct_triattention_0.75 | 800 | 0 | 38.74 | 0.355 |
| Llama-3.1-8B-Instruct_triattention_0.8 | 800 | 0 | 37.79 | 0.288 |
| Llama-3.1-8B-Instruct_triattention_0.9 | 800 | 0 | 33.10 | 0.154 |
| Llama-3.1-8B-Instruct_triattention_0.95 | 800 | 0 | 31.41 | 0.087 |
| Llama-3.1-8B-Instruct_turboquant_2.5bit_0.0 | 800 | 0 | 39.89 | 0.262 |
| Llama-3.1-8B-Instruct_turboquant_2bit_0.0 | 800 | 0 | 14.94 | 0.204 |
| Llama-3.1-8B-Instruct_turboquant_3.5bit_0.0 | 800 | 0 | 47.25 | 0.346 |
| Llama-3.1-8B-Instruct_turboquant_3bit_0.0 | 800 | 0 | 44.48 | 0.288 |
| Llama-3.1-8B-Instruct_turboquant_4bit_0.0 | 800 | 0 | 47.67 | 0.372 |
| Llama-3.1-8B-Instruct_turboquant_adaptive_0.0 | 800 | 0 | 44.57 | 0.288 |
| Llama-3.1-8B-Instruct_turboquant_mse_2.5bit_0.0 | 800 | 0 | 46.44 | 0.251 |
| Llama-3.1-8B-Instruct_turboquant_mse_3.5bit_0.0 | 800 | 0 | 47.84 | 0.335 |
| Llama-3.1-8B-Instruct_turboquant_mse_3bit_0.0 | 800 | 0 | 47.90 | 0.283 |
| Llama-3.1-8B-Instruct_turboquant_mse_4bit_0.0 | 800 | 0 | 47.57 | 0.366 |

## Checks: 39 passed, 0 failed

| Stage | Check | Expected | Observed | Status |
|---|---|---|---|---|
| ingestion | raw record count | 3750 | 3750 | PASS |
| ingestion | unique raw IDs | 3750 | 3750 | PASS |
| cleaning | clean record count | 3731 | 3731 | PASS |
| cleaning | rows removed by cleaning | 19 | 19 | PASS |
| cleaning | clean IDs all come from raw | 0 | 0 | PASS |
| transformation | prompts over the 32,000-token cap | 0 | 0 | PASS |
| transformation | truncated prompts | 88 | 88 | PASS |
| preparation | record IDs in more than one split | 0 | 0 | PASS |
| preparation | train rows | 2131 | 2131 | PASS |
| preparation | validation rows | 800 | 800 | PASS |
| preparation | test rows | 800 | 800 | PASS |
| preparation | split IDs missing from clean data | 0 | 0 | PASS |
| preparation | records per task in validation | 50 x 16 | 50 x 16 | PASS |
| preparation | records per task in test | 50 x 16 | 50 x 16 | PASS |
| preparation | test records sharing a document with validation (disclosed list) | 51 | 51 | PASS |
| results | Llama-3.1-8B-Instruct_turboquant_adaptive_0.0: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_turboquant_2.5bit_0.0: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_turboquant_2bit_0.0: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_turboquant_3.5bit_0.0: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_turboquant_3bit_0.0: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_turboquant_4bit_0.0: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_turboquant_mse_2.5bit_0.0: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_turboquant_mse_3.5bit_0.0: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_turboquant_mse_3bit_0.0: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_turboquant_mse_4bit_0.0: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_none_0.0: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_kvzip_0.75: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_kvzip_0.9: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_restorekv_0.75: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_restorekv_0.8: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_restorekv_0.9: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_restorekv_0.95: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_restorekv_4bit_0.95: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_snapkv_0.75: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_snapkv_0.9: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_triattention_0.75: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_triattention_0.8: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_triattention_0.9: records outside validation | 0 | 0 | PASS |
| results | Llama-3.1-8B-Instruct_triattention_0.95: records outside validation | 0 | 0 | PASS |
