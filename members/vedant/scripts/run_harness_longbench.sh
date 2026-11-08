#!/usr/bin/env bash
# TurboQuant on the shared harness: the team's 800 dev records, prompts, BF16 model,
# scorer and kv_cache_gb, with the question and answer left uncompressed.
#
#   cd /app/298a-kv-cache-compression
#   hf auth login          # the account needs access to meta-llama/Llama-3.1-8B-Instruct
#   bash members/vedant/scripts/run_harness_longbench.sh 2>&1 | tee -a turboquant.log
#
# Nothing to install: the lab image already has kvpress 0.5.5 and transformers 5.2.0.
# The dev set is not in git: unzip the team's processed_longbench.zip so that
# data/processed_longbench/processed_longbench_dev.jsonl exists at the repo root.
# Each run saves after every record and skips saved records, so if the machine stops,
# start the same command again. Results go to results/longbench/ as
# Llama-3.1-8B-Instruct_turboquant_<setting>_0.0.jsonl, next to the 47.72 full-cache run.

set -u
cd "$(dirname "$0")/../../.."

DATA=data/processed_longbench/processed_longbench_dev.jsonl
if [ ! -f "$DATA" ]; then
    echo "Missing $DATA: unzip the team's processed_longbench.zip at the repo root first." >&2
    exit 1
fi

run() {
    echo "=== $(date '+%a %H:%M') $*"
    python "$@" || echo "!!! failed: $*"
}

# 0. Two records per task (about 5 minutes). Stops here if it fails.
echo "=== $(date '+%a %H:%M') check: turboquant_3bit, 2 records per task"
python harness/run_longbench.py --press turboquant_3bit --per-task 2 || {
    echo "!!! check failed: fix this before the full runs" >&2
    exit 1
}

# 1. Main settings, most important first: the paper's TurboQuant (QJL keys) at 3 and 4 bits,
#    then the same bit widths without QJL, then 2 bits.
for press in turboquant_3bit turboquant_4bit turboquant_mse_3bit turboquant_mse_4bit turboquant_2bit; do
    run harness/run_longbench.py --press "$press"
done

echo "=== $(date '+%a %H:%M') TurboQuant runs finished"
