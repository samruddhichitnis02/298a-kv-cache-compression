#!/usr/bin/env bash
# Re-run TriAttention with a proper calibration (lab RTX 5090, about 4.5 hours).
#
# The Nov 10 runs were calibrated on five sentences repeated to 4,096 tokens, which the
# authors' calibration guide says to avoid. This recalibrates on 32,768 tokens of natural
# LongBench train-split text and reruns the four ratios. Nothing else changes: same press,
# same 800 records, same scorer. The old results stay in results/triattention/longbench/.
#
#   nohup bash members/anees/run_triattention_recal.sh > triattention_recal.log 2>&1 &
#
# Safe to restart after a shutdown: calibration is skipped if the stats file exists, and
# the harness skips records already written.
set -euo pipefail
cd "$(dirname "$0")/../.."

STATS=members/samruddhi/calibration/llama31_8b_stats_trainsplit.pt
OUT=results/triattention/longbench_trainsplit_calibration

if [ ! -f "$STATS" ]; then
  echo "=== calibration $(date)"
  python members/samruddhi/calibrate_triattention.py --output "$STATS"
fi

for ratio in 0.75 0.9 0.8 0.95; do
  echo "=== triattention $ratio $(date)"
  python harness/run_longbench.py --press triattention --compression-ratio "$ratio" \
    --triattention-stats "$STATS" --output-dir "$OUT"
done
echo "=== done $(date)"
