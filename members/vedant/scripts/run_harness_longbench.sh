#!/usr/bin/env bash
# TurboQuant on the shared harness: the team's 800 dev records, prompts, BF16 model,
# scorer and kv_cache_gb, with the question and answer left uncompressed.
#
#   cd /app/298a-kv-cache-compression
#   hf auth login
#   bash members/vedant/scripts/run_harness_longbench.sh 2>&1 | tee -a turboquant.log
#
# Nothing to install: the lab image already has kvpress 0.5.5 and transformers 5.2.0.
# Each run saves after every record and skips saved records, so if the machine stops,
# start the same command again. Results go to results/longbench/ as
# Llama-3.1-8B-Instruct_turboquant_<setting>_0.0.jsonl, next to the 47.72 full-cache run.

set -u
cd "$(dirname "$0")/../../.."

run() {
    echo "=== $(date '+%a %H:%M') $*"
    python "$@" || echo "!!! failed: $*"
}

# 0. Two records per task (about 5 minutes). Stop and check the log if this fails.
run harness/run_longbench.py --press turboquant_3bit --per-task 2

# 1. Main settings, most important first: the paper's TurboQuant (QJL keys) at 3 and 4 bits,
#    then the same bit widths without QJL, then 2 bits.
for press in turboquant_3bit turboquant_4bit turboquant_mse_3bit turboquant_mse_4bit turboquant_2bit; do
    run harness/run_longbench.py --press "$press"
done

echo "=== $(date '+%a %H:%M') TurboQuant runs finished"
