#!/usr/bin/env bash
# Quick launcher for 03d_sam3_boundary_reconstruction.py
# Edit the variables below, then: bash run_03d.sh
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PROJECT_DIR/external/sam3:$PYTHONPATH"

# ── edit these ────────────────────────────────────────────────────────────────
FRAMES_DIR="//home/chengzhe/Data/OMS_data3/rs_bags/20260127_015119_20260415_010454"
TRAJECTORY="$PROJECT_DIR/output/20260127_015119_20260415_010454/sparse/trajectory_open3d.log"
OUTPUT_DIR="$PROJECT_DIR/output/gamma_0.3"

VOXEL_SIZE=0.005
# Space-separated list — each value produces a thresh_<N>/ subdirectory.
# EDT and TSDF run once; only culling + segmentation repeat per threshold.
ALPHA_THRESHOLDS="0.3 0.4 0.5 0.6"
EDT_GAMMA=0.3
SAM3_MAX_SIZE_RATIO=0.15
MESH_KEEP_COMPONENTS=1
BOUNDARY_PROPAGATION_HOPS=0
MIN_CLUSTER_SIZE=1000
# ─────────────────────────────────────────────────────────────────────────────

conda run -n slam_recon --no-capture-output \
python -u "$PROJECT_DIR/scripts/03d_sam3_boundary_reconstruction.py" \
    --frames_dir                "$FRAMES_DIR"                        \
    --intrinsic                 "$FRAMES_DIR/intrinsic.json"         \
    --trajectory                "$TRAJECTORY"                        \
    --output                    "$OUTPUT_DIR/raw_mesh_rgb.ply"       \
    --voxel_size                "$VOXEL_SIZE"                        \
    --alpha_thresholds          $ALPHA_THRESHOLDS                    \
    --edt_gamma                 "$EDT_GAMMA"                         \
    --sam_max_size_ratio        "$SAM3_MAX_SIZE_RATIO"               \
    --mesh_keep_components      "$MESH_KEEP_COMPONENTS"              \
    --boundary_propagation_hops "$BOUNDARY_PROPAGATION_HOPS"         \
    --min_cluster_size          "$MIN_CLUSTER_SIZE"
