# Point 1: hybrid eviction + quantization at measured bytes

From the result files in `results/hybrid/longbench`.

| Method | Strategy | Records | Measured document cache | Score |
|---|---|---|---|---|
| SnapKV | hybrid, 4-bit | 400 | 10.0% | 41.90 |
| TriAttention | hybrid, 4-bit | 400 | 10.0% | 42.30 |
| SnapKV | hybrid, 2-bit | 400 | 10.0% | 43.81 |
| TriAttention | hybrid, 2-bit | 400 | 10.0% | 41.87 |
| SnapKV | eviction only | 400 | 10.0% | 34.69 |
| TriAttention | eviction only | 400 | 10.0% | 34.98 |
| RDKV | eviction only (bit-widths 0 and 16) | 400 | 10.0% | 35.90 |
| RestoreKV | hybrid, 4-bit | 400 | 10.0% | 48.74 |
| KVzip | hybrid, 2-bit | 400 | 10.0% | 45.85 |
| RestoreKV | hybrid, 2-bit | 400 | 10.0% | 47.25 |
| KVzip | hybrid, 4-bit | 400 | 10.0% | 50.00 |
| KVzip | eviction only | 400 | 10.0% | 43.53 |
| RestoreKV | eviction only | 400 | 10.0% | 44.57 |
| RDKV | hybrid, mixed bit-widths | 400 | 10.3% | 43.00 |
| Full cache | quantization only, 2-bit | 400 | 18.8% | 46.93 |
| Full cache | quantization only, 4-bit | 400 | 31.3% | 48.61 |
| Full cache | none (baseline run, same records) | 400 | 100.0% | 48.66 |
