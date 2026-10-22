#!/usr/bin/env bash
# 07_export_rhino.sh — Export camera cones + trajectory (cameras.obj) for Rhino,
# into <run dir>/export/. Import the run's PLYs alongside it: same SLAM world
# frame, metres.
# Settings: pipeline.yaml (paths.*, export.*). Run:  bash 07_export_rhino.sh
set -e

# ── OVERRIDES — this run only; uncomment to replace pipeline.yaml values ──────
# EXPORT_DIR=/abs/path/to/export
# EXPORT_CONE_EVERY=30
# ──────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/scripts/pipeline_env.sh"

python "$PROJECT_DIR/scripts/07_export_rhino.py" \
    --frames_dir  "$FRAMES_DIR"        \
    --trajectory  "$TRAJECTORY"        \
    --output_dir  "$EXPORT_DIR"        \
    --cone_depth  "$EXPORT_CONE_DEPTH" \
    --cone_every  "$EXPORT_CONE_EVERY"
