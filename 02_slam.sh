#!/usr/bin/env bash
# 02_slam.sh — Run ORB-SLAM3 RGB-D tracking and convert the trajectory.
# Saves the Atlas so other sensors can relocalize against this map.
# Edit the variables below, then run:  bash 02_slam.sh
set -e

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh
conda activate slam_recon

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── EDIT THESE ────────────────────────────────────────────────────────────────
FRAMES_DIR=/home/chengzhe/Data/OMS_data3/rs_bags/base_20260127_015119
OUTPUT_DIR=$PROJECT_DIR/output/base/sparse
FPS=30            # effective FPS of the extracted frames

# Atlas path (no .osa extension) — set to empty string "" to skip saving
SAVE_ATLAS=$OUTPUT_DIR/atlas

# Uncomment to load an existing atlas (localization mode):
# LOAD_ATLAS=$OUTPUT_DIR/atlas
# LOCALIZE=--localize
# ──────────────────────────────────────────────────────────────────────────────

ATLAS_ARG=""
[ -n "${SAVE_ATLAS:-}" ] && ATLAS_ARG="--save_atlas $SAVE_ATLAS"

# LOAD_ATLAS_ARG=""
# [ -n "${LOAD_ATLAS:-}" ] && LOAD_ATLAS_ARG="--load_atlas $LOAD_ATLAS"

python "$PROJECT_DIR/scripts/02_slam.py" \
    --frames_dir "$FRAMES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --fps        "$FPS"        \
    --headless                 \
    $ATLAS_ARG                 \
    ${LOAD_ATLAS_ARG:-}        \
    ${LOCALIZE:-}
