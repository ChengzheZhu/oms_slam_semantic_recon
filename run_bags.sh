#!/usr/bin/env bash
# run_bags.sh — Run a step range of the pipeline for several bags in a row.
#
# Each bag runs run_pipeline.sh with PATHS_BAG set to it; everything else comes
# from pipeline.yaml or the environment (e.g. CULL_SCORE_THRESHOLDS). A failed
# bag is logged and the batch moves on to the next one.
#
# Usage:
#   bash run_bags.sh FIRST LAST BAG...
#   CULL_SCORE_THRESHOLDS="0.3 0.5" bash run_bags.sh 4 6 /data/a.bag /data/b.bag
#
# Logs: <run dir>/run_<FIRST>_<LAST>.log per bag; summary on stdout.

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -lt 3 ]; then
    sed -n '2,12p' "$0"; exit 1
fi
FIRST=$1; LAST=$2; shift 2

failed=()
for bag in "$@"; do
    run_name="$(basename "$bag" .bag)"
    log_dir="$PROJECT_DIR/output/$run_name"
    mkdir -p "$log_dir"
    log="$log_dir/run_$(printf '%02d_%02d' "$FIRST" "$LAST").log"
    echo "[$(date '+%F %T')] START $run_name  steps $FIRST–$LAST  log $log"
    if PATHS_BAG="$bag" bash "$PROJECT_DIR/run_pipeline.sh" "$FIRST" "$LAST" > "$log" 2>&1; then
        echo "[$(date '+%F %T')] DONE  $run_name"
    else
        echo "[$(date '+%F %T')] FAIL  $run_name  (exit $?)  see $log"
        failed+=("$run_name")
    fi
done

echo "[$(date '+%F %T')] BATCH END  $(( $# - ${#failed[@]} ))/$# ok${failed:+  failed: ${failed[*]}}"
[ ${#failed[@]} -eq 0 ]
