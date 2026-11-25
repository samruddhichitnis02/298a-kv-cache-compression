#!/usr/bin/env bash
# One command per review point. Each owner runs their point across the methods.
#
#   bash harness/sprint/run_point.sh timing smoke      # a few short records first, about 10-20 minutes
#   bash harness/sprint/run_point.sh timing            # the real runs
#
# Points: hybrid (point 1), timing (point 3), many (point 4, many questions), long (point 4, long documents).
# Every run saves after each record and skips what is already saved, so after a shutdown
# run the same command again. A run that fails is reported and the script moves on.
#
#   nohup bash harness/sprint/run_point.sh hybrid > hybrid.log 2>&1 &     # keeps going if the terminal closes
#   FULL=1 bash harness/sprint/run_point.sh hybrid                        # all 800 records instead of 400
set -u
cd "$(dirname "$0")/../.."

POINT="${1:-}"
SMOKE="${2:-}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
FULL_DATA=data/processed_longbench/processed_longbench.jsonl
STATS=members/samruddhi/calibration/llama31_8b_stats_trainsplit.pt
FAILED=0

run() {
    echo "=== $(date '+%a %H:%M') python $*"
    python "$@" || { echo "!!! FAILED: python $*"; FAILED=$((FAILED + 1)); }
}

# TriAttention uses the recalibrated statistics once Samruddhi's rerun has produced them.
tri_stats() { if [ -f "$STATS" ]; then echo "--triattention-stats $STATS"; fi; }

longbench() {   # longbench <output folder> <args...>
    local out="$1"; shift
    if [ "$SMOKE" = "smoke" ]; then
        run harness/run_longbench.py --model "$MODEL" --tasks qasper,lcc --per-task 1 --max-prompt-tokens 8000 \
            --timing-repeats 1 --output-dir "results/smoke/$out" "$@"
    else
        run harness/run_longbench.py --model "$MODEL" --output-dir "results/$out" "$@"
    fi
}

needles() {     # needles <output folder> <length> <args...>
    local out="$1" length="$2"; shift 2
    if [ "$SMOKE" = "smoke" ]; then
        run harness/run_needles.py --model "$MODEL" --length 4096 --per-task 2 --output-dir "results/smoke/$out" "$@"
    else
        run harness/run_needles.py --model "$MODEL" --length "$length" --output-dir "results/$out" "$@"
    fi
}

make_documents() {   # make_documents <length...>
    if [ "$SMOKE" = "smoke" ]; then set -- 4096; fi
    for length in "$@"; do
        [ -f "data/needles/needles_${length}.jsonl" ] || run harness/needles.py --length "$length" --tokenizer "$MODEL"
    done
}

case "$POINT" in
hybrid)
    # Point 1. Eviction only, hybrid at 4 bits and hybrid at 2 bits, all at 10% of the full document
    # cache in measured bytes; then quantization only. 400 records (25 per task) unless FULL=1.
    SIZE="--per-task 25"; [ "${FULL:-}" = "1" ] && SIZE=""
    [ "$SMOKE" = "smoke" ] && SIZE=""     # the smoke run sets its own --per-task 1
    for press in restorekv kvzip snapkv triattention; do
        extra=""; [ "$press" = "triattention" ] && extra="$(tri_stats)"
        for bits in 4 16 2; do
            longbench hybrid/longbench --press "$press" --pack-bits "$bits" --memory-budget 0.1 $SIZE $extra
        done
    done
    longbench hybrid/longbench --press none --pack-bits 4 $SIZE
    longbench hybrid/longbench --press none --pack-bits 2 $SIZE
    # RDKV mixes bit-widths, so it has its own packing (members/manjot/rdkv_pack.py): the hybrid, then its
    # eviction-only half (bit-widths 0 and 16 only) at the same budget.
    longbench hybrid/longbench --press rdkv --compression-ratio 0.9 --rdkv-packed $SIZE
    longbench hybrid/longbench --press rdkv --compression-ratio 0.9 --rdkv-bitset 0,16 --rdkv-packed $SIZE
    ;;
timing)
    # Point 3. The same 100 records for every method, 3 repeats each, full cache first.
    REC="--records harness/subsets/timing_100.json"; [ "$SMOKE" = "smoke" ] && REC=""
    longbench timing/longbench --press none --timing $REC
    for press in snapkv kvzip restorekv rdkv triattention; do
        extra=""; [ "$press" = "triattention" ] && extra="$(tri_stats)"
        longbench timing/longbench --press "$press" --compression-ratio 0.9 --timing $REC $extra
    done
    longbench timing/longbench --press restorekv --pack-bits 4 --memory-budget 0.1 --timing $REC
    longbench timing/longbench --press turboquant_mse_3bit --timing $REC
    ;;
many)
    # Point 4, many questions. 16 questions answered from one compressed document, at 16K and 32K.
    make_documents 16384 32768
    for length in 16384 32768; do
        needles many "$length" --tasks many --press none
        for press in restorekv kvzip snapkv rdkv triattention; do
            extra=""; [ "$press" = "triattention" ] && extra="$(tri_stats)"
            needles many "$length" --tasks many --press "$press" --compression-ratio 0.9 $extra
        done
        # what a question-aware method loses on later questions
        needles many "$length" --tasks many --press snapkv --compression-ratio 0.9 --compress-with-first-question
        [ "$SMOKE" = "smoke" ] && break
    done
    ;;
long)
    # Point 4, long documents. 32K and 64K; chunked prefill where the method allows it.
    make_documents 32768 65536
    TASKS="--tasks single,multikey,multivalue,multiquery"
    for length in 32768 65536; do
        needles long "$length" $TASKS --press none --chunk-prefill 8192
        for press in restorekv kvzip; do
            needles long "$length" $TASKS --press "$press" --compression-ratio 0.9 --chunk-prefill 8192
        done
        for press in snapkv rdkv triattention; do     # one pass only; at 64K these may run out of memory
            extra=""; [ "$press" = "triattention" ] && extra="$(tri_stats)"
            needles long "$length" $TASKS --press "$press" --compression-ratio 0.9 $extra
        done
        [ "$SMOKE" = "smoke" ] && break
    done
    ;;
*)
    echo "usage: bash harness/sprint/run_point.sh {hybrid|timing|many|long} [smoke]"; exit 2 ;;
esac

echo "=== $(date '+%a %H:%M') finished: $POINT ${SMOKE} with $FAILED failed run(s)"
[ "$FAILED" -eq 0 ]
