#!/usr/bin/env bash
# 01_extract.sh — Extract RGB-D frames from a RealSense .bag file.
# Edit the variables below, then run:  bash 01_extract.sh
set -e

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh
conda activate slam_recon

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── EDIT THESE ────────────────────────────────────────────────────────────────
BAG=/home/chengzhe/Data/OMS_data3/rs_bags/20260127_015119.bag
OUTPUT=/home/chengzhe/Data/OMS_data3/rs_bags/base_20260127_015119
STRIDE=1          # extract every Nth frame (1 = all)
MAX_FRAMES=0      # 0 = no limit
# ──────────────────────────────────────────────────────────────────────────────

python "$PROJECT_DIR/scripts/01_extract_frames.py" \
    --bag        "$BAG"        \
    --output     "$OUTPUT"     \
    --stride     "$STRIDE"     \
    --max_frames "$MAX_FRAMES"
