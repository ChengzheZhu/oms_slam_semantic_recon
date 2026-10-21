#!/usr/bin/env bash
# 01_extract.sh — Extract RGB-D frames from a RealSense .bag file.
# Settings: pipeline.yaml (paths.*, extract.*). Run:  bash 01_extract.sh
set -e

# ── OVERRIDES — this run only; uncomment to replace pipeline.yaml values ──────
# PATHS_BAG=/abs/path/to/other.bag
# FRAMES_DIR=/abs/path/to/frames
# EXTRACT_STRIDE=5
# EXTRACT_MAX_FRAMES=300
# ──────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/scripts/pipeline_env.sh"

python "$PROJECT_DIR/scripts/01_extract_frames.py" \
    --bag        "$PATHS_BAG"          \
    --output     "$FRAMES_DIR"         \
    --stride     "$EXTRACT_STRIDE"     \
    --max_frames "$EXTRACT_MAX_FRAMES" \
    $(flag "$EXTRACT_SKIP_CONFIDENCE" --skip_confidence)
