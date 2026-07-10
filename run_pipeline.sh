#!/usr/bin/env bash
# run_pipeline.sh — Run the full pipeline steps 01–06 in sequence.
#
# This script is a thin orchestrator. All parameters live in the individual
# step scripts (01_extract.sh … 06_cull_segment.sh). Edit those first.
#
# Usage:
#   bash run_pipeline.sh              # run all steps
#   bash run_pipeline.sh 3 6          # run steps 03 through 06 only
#
# Steps:
#   01  Extract frames from .bag
#   02  ORB-SLAM3 tracking + trajectory conversion
#   03  RGB TSDF meshing
#   04  SAM3 mask cache (L1)
#   05  SAM3 EDT scoring + semantic TSDF (L2)
#   06  Cull seams + segment stones
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

FIRST_STEP=${1:-1}
LAST_STEP=${2:-6}

run_step() {
    local n=$1
    local script=$2
    if [ "$n" -ge "$FIRST_STEP" ] && [ "$n" -le "$LAST_STEP" ]; then
        echo ""
        echo "════════════════════════════════════════"
        echo " Step 0$n — $(basename "$script" .sh)"
        echo "════════════════════════════════════════"
        bash "$script"
    fi
}

run_step 1 "$PROJECT_DIR/01_extract.sh"
run_step 2 "$PROJECT_DIR/02_slam.sh"
run_step 3 "$PROJECT_DIR/03_tsdf_rgb.sh"
run_step 4 "$PROJECT_DIR/04_sam3_mask.sh"
run_step 5 "$PROJECT_DIR/05_sam3_score.sh"
run_step 6 "$PROJECT_DIR/06_cull_segment.sh"

echo ""
echo "════════════════════════════════════════"
echo " Pipeline complete (steps $FIRST_STEP–$LAST_STEP)"
echo "════════════════════════════════════════"
