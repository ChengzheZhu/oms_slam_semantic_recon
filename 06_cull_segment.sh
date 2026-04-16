#!/usr/bin/env bash
# 06_cull_segment.sh — Transfer scores, cull seam triangles, segment stones.
# Sweep multiple alpha thresholds in one run; each produces a thresh_*/ subdir.
# Use --skip_segments for quick threshold tuning (no per-stone PLY export).
# Edit the variables below, then run:  bash 06_cull_segment.sh
set -e

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh
conda activate slam_recon

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET=base

# ── EDIT THESE ────────────────────────────────────────────────────────────────
RAW_MESH=$PROJECT_DIR/output/$DATASET/raw_mesh_rgb.ply
ALPHA_MESH=$PROJECT_DIR/output/$DATASET/scoring/alpha_mesh.ply
OUTPUT_DIR=$PROJECT_DIR/output/$DATASET/segments

# Space-separated list; each value → thresh_<t>/ subdir
ALPHA_THRESHOLDS="0.5"

MESH_KEEP_COMPONENTS=1   # keep N largest components (0 = disabled)
BOUNDARY_HOPS=0          # BFS hops to fatten seam network (0 = disabled)
MIN_CLUSTER_SIZE=1000    # minimum triangles per saved segment

# Uncomment to skip per-stone PLY export (threshold tuning only):
# SKIP_SEGMENTS=--skip_segments
# ──────────────────────────────────────────────────────────────────────────────

python "$PROJECT_DIR/scripts/06_cull_segment.py" \
    --raw_mesh                  "$RAW_MESH"                  \
    --alpha_mesh                "$ALPHA_MESH"                \
    --output_dir                "$OUTPUT_DIR"                \
    --alpha_thresholds          $ALPHA_THRESHOLDS            \
    --mesh_keep_components      "$MESH_KEEP_COMPONENTS"      \
    --boundary_propagation_hops "$BOUNDARY_HOPS"             \
    --min_cluster_size          "$MIN_CLUSTER_SIZE"          \
    ${SKIP_SEGMENTS:-}
