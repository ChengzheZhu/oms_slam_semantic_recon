#!/usr/bin/env bash
# 06_cull_segment.sh — Transfer scores, cull seam triangles, segment stones.
# Sweep multiple alpha thresholds in one run; each produces a thresh_*/ subdir.
# Set cull.skip_segments: true for quick threshold tuning (no per-stone PLY export).
# Settings: pipeline.yaml (paths.*, cull.*). Run:  bash 06_cull_segment.sh
set -e

# ── OVERRIDES — this run only; uncomment to replace pipeline.yaml values ──────
# RAW_MESH=/abs/path/to/raw_mesh_rgb.ply
# SEGMENTS_DIR=/abs/path/to/segments
# CULL_ALPHA_THRESHOLDS="0.3 0.35 0.4"
# CULL_SKIP_SEGMENTS=false
# ──────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/scripts/pipeline_env.sh"

python "$PROJECT_DIR/scripts/06_cull_segment.py" \
    --raw_mesh                  "$RAW_MESH"                   \
    --alpha_mesh                "$SCORING_DIR/alpha_mesh.ply" \
    --output_dir                "$SEGMENTS_DIR"               \
    --alpha_thresholds          $CULL_ALPHA_THRESHOLDS        \
    --mesh_keep_components      "$CULL_MESH_KEEP_COMPONENTS"  \
    --boundary_propagation_hops "$CULL_BOUNDARY_HOPS"         \
    --min_cluster_size          "$CULL_MIN_CLUSTER_SIZE"      \
    $(flag "$CULL_SKIP_SEGMENTS" --skip_segments)
