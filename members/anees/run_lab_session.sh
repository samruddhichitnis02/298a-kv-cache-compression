#!/usr/bin/env bash
# Anees's lab session for the review points: check the shared tools on the GPU first, then point 2.
#
#   bash members/anees/run_lab_session.sh smoke      # 1. smoke-run all four points (about 1 to 1.5 hours)
#   bash members/anees/run_lab_session.sh 64k        # 2. does a 64K document fit on this GPU? (a few minutes)
#   bash members/anees/run_lab_session.sh adaptive   # 3. fit the adaptive rule on training records, test on validation (about 4.5 hours)
#   bash members/anees/run_lab_session.sh all        # everything, in that order
#
#   nohup bash members/anees/run_lab_session.sh all > lab_session.log 2>&1 &     # keeps going if the terminal closes
#
# Every run saves after each record and skips what is saved, so after a shutdown run the same command
# again. A step that fails is reported and the script moves on.
set -u
cd "$(dirname "$0")/../.."

STEP="${1:-}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PRESS="${PRESS:-restorekv}"        # the method the adaptive rule is fitted for
TARGET="${TARGET:-0.9}"            # the fixed ratio whose memory is the budget
EXTRA="${EXTRA:-}"                 # extra run_longbench.py arguments, for quick checks only
FULL_DATA=data/processed_longbench/processed_longbench.jsonl
FAILED=0

run() {
    echo "=== $(date '+%a %H:%M') $*"
    "$@" || { echo "!!! FAILED: $*"; FAILED=$((FAILED + 1)); }
}

smoke() {
    for point in timing hybrid many long; do
        echo "##### smoke: $point"
        MODEL="$MODEL" bash harness/sprint/run_point.sh "$point" smoke 2>&1 | tee "smoke_${point}.log"
        grep -q "with 0 failed run(s)" "smoke_${point}.log" || FAILED=$((FAILED + 1))
    done
    echo "##### smoke summary"
    grep -h -E "^!!! FAILED|finished:" smoke_*.log
}

check_64k() {
    # Three 64K documents: the full cache with chunked prefill, then the two methods that can be chunked.
    [ -f data/needles/needles_65536.jsonl ] || run python harness/needles.py --length 65536 --tokenizer "$MODEL"
    for press in none restorekv kvzip; do
        ratio=""; [ "$press" != "none" ] && ratio="--compression-ratio 0.9"
        run python harness/run_needles.py --model "$MODEL" --length 65536 --tasks single --per-task 3 \
            --press "$press" $ratio --chunk-prefill 8192 --output-dir results/smoke/long64k
    done
    echo "##### 64K summary (peak GPU memory in GB, out-of-memory count)"
    python - <<'PY'
import glob, json
for path in sorted(glob.glob("results/smoke/long64k/*.jsonl")):
    rows = [json.loads(line) for line in open(path)]
    peaks = [r["peak_gpu_gb"] for r in rows if r.get("peak_gpu_gb")]
    print(f"{path.split('/')[-1]:60s} documents {len(rows)}  oom {sum(r['oom'] for r in rows)}  "
          f"peak {max(peaks) if peaks else 'n/a'}  score {sum(sum(r['scores']) / len(r['scores']) for r in rows) / len(rows):.2f}")
PY
}

adaptive() {
    # Review point 2, the clean test: fit the rule on training records only, then run all 800 validation records.
    train="results/$PRESS/train_sample"
    for ratio in 0.9 0.75 0.95 0.8; do
        run python harness/run_longbench.py --model "$MODEL" --press "$PRESS" --compression-ratio "$ratio" \
            --data "$FULL_DATA" --records harness/subsets/train_300.json --output-dir "$train" $EXTRA
    done
    policy="results/$PRESS/adaptive_policy_${TARGET}.json"
    run python harness/adaptive.py fit --target "$TARGET" --runs "$train"/*.jsonl --data "$FULL_DATA" --out "$policy"
    echo "##### how the rule does on the training records themselves (held out within them)"
    run python harness/adaptive.py evaluate --target "$TARGET" --runs "$train"/*.jsonl --data "$FULL_DATA" --repeats 20
    run python harness/run_longbench.py --model "$MODEL" --press "$PRESS" --adaptive-policy "$policy" \
        --output-dir "results/$PRESS/adaptive" $EXTRA
    echo "##### adaptive against the fixed ratio on the validation records"
    python - "$PRESS" "$TARGET" "${MODEL##*/}" <<'PY'
import glob, json, sys
press, target, model = sys.argv[1], sys.argv[2], sys.argv[3]

def load(pattern):
    paths = glob.glob(pattern)
    return {r["record_id"]: r for r in map(json.loads, open(paths[0]))} if paths else None

def macro(rows, ids):
    tasks = {}
    for i in ids:
        tasks.setdefault(rows[i]["task"], []).append(rows[i]["score"])
    return 100 * sum(sum(v) / len(v) for v in tasks.values()) / len(tasks)

adaptive = load(f"results/{press}/adaptive/{model}_{press}_adaptive{target}.jsonl")
fixed = load(f"results/*/longbench/{model}_{press}_{target}.jsonl")
if not adaptive or not fixed:
    sys.exit("adaptive or fixed result file not found; the fixed runs are in results/<method>/longbench/")
ids = sorted(set(adaptive) & set(fixed))
gb = lambda rows: sum(rows[i]["kv_cache_gb"] for i in ids) / len(ids)
print(f"records {len(ids)}  fixed {macro(fixed, ids):.2f} at {gb(fixed):.4f} GB  "
      f"adaptive {macro(adaptive, ids):.2f} at {gb(adaptive):.4f} GB  "
      f"gain {macro(adaptive, ids) - macro(fixed, ids):+.2f}  memory {100 * gb(adaptive) / gb(fixed):.0f}% of fixed")
PY
}

case "$STEP" in
smoke) smoke ;;
64k) check_64k ;;
adaptive) adaptive ;;
all) smoke; check_64k; adaptive ;;
*) echo "usage: bash members/anees/run_lab_session.sh {smoke|64k|adaptive|all}"; exit 2 ;;
esac
echo "=== $(date '+%a %H:%M') finished: $STEP with $FAILED failed step(s)"
[ "$FAILED" -eq 0 ]
