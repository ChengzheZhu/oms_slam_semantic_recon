#!/usr/bin/env bash
# 03_tsdf_rgb.sh — Fuse RGB-D frames into a dense coloured mesh (TSDF).
# Settings: pipeline.yaml (paths.*, tsdf.*). Run:  bash 03_tsdf_rgb.sh
set -e

# ── OVERRIDES — this run only; uncomment to replace pipeline.yaml values ──────
# RAW_MESH=/abs/path/to/raw_mesh_rgb.ply
# TSDF_VOXEL_SIZE=0.005
# TSDF_DEPTH_MAX=2.0
# TSDF_CONFIDENCE_THRESHOLD=1
# ──────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/scripts/pipeline_env.sh"

python "$PROJECT_DIR/scripts/03_tsdf_rgb.py" \
    --frames_dir            "$FRAMES_DIR"                \
    --trajectory            "$TRAJECTORY"                \
    --output                "$RAW_MESH"                  \
    --voxel_size            "$TSDF_VOXEL_SIZE"           \
    --depth_max             "$TSDF_DEPTH_MAX"            \
    --depth_min             "$TSDF_DEPTH_MIN"            \
    --confidence_threshold  "$TSDF_CONFIDENCE_THRESHOLD"
