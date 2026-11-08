#!/usr/bin/env bash
# All of Anees's remaining RTX 5090 runs, most important first (SAM-95, SAM-96, SAM-97).
#
#   cd /app/298a-kv-cache-compression
#   bash members/anees/run_queue.sh 2>&1 | tee -a queue.log
#
# Each run saves after every record and skips records already saved, so if the
# machine shuts down, start the same command again and it carries on.
# Rough times on the 5090: KVzip and quantized RestoreKV about 1.6-1.8 h per
# LongBench ratio, SnapKV about 0.5 h. Groups 1-3 take about 15 h, group 4 about 2.5 h.

set -u
cd "$(dirname "$0")/../.."

run() {
    echo "=== $(date '+%a %H:%M') $*"
    python "$@" || echo "!!! failed: $*"
}

# 0. Ten-minute check that Charvee's ResidualKV runs on the lab image, before her own
#    lab slot: 2 records per task. Needs only a clone of the authors' code, no installs.
if [ ! -d ../DeltaKV ]; then
    git clone -q https://github.com/CURRENTF/DeltaKV ../DeltaKV &&
        git -C ../DeltaKV checkout -q 11a837b21acfa475957210c5b2d339b89c1b7e3b
fi
run members/charvee/run_residualkv.py --compression-ratio 0.9 --per-task 2

# 1. Comparison rows for the LongBench table (SAM-95), main budgets first
for ratio in 0.9 0.75; do
    run harness/run_longbench.py --press kvzip --compression-ratio "$ratio"
    run harness/run_longbench.py --press snapkv --compression-ratio "$ratio"
done

# 2. RestoreKV + quantization at 5% and 10% memory budgets (SAM-96)
for ratio in 0.95 0.9; do
    run harness/run_longbench.py --press restorekv_4bit --compression-ratio "$ratio"
    run harness/run_longbench.py --press restorekv_2bit --compression-ratio "$ratio"
done

# 3. Remaining comparison budgets
for ratio in 0.95 0.8; do
    run harness/run_longbench.py --press kvzip --compression-ratio "$ratio"
    run harness/run_longbench.py --press snapkv --compression-ratio "$ratio"
done

# 4. Follow-up questions on one compressed cache, 125 documents, 10% budget (SAM-98).
#    Needs the full processed set, data/processed_longbench/processed_longbench.jsonl
for press in none snapkv kvzip restorekv; do
    run members/anees/run_followup.py --press "$press" --compression-ratio 0.9
done

# 5. Length sweep on RULER at a 5% budget (SAM-97), if time is left
for length in 8192 16384; do
    for press in none kvzip restorekv; do
        run harness/run_ruler.py --length "$length" --press "$press" --compression-ratio 0.95 --per-task 25
    done
done

echo "=== $(date '+%a %H:%M') queue finished"
