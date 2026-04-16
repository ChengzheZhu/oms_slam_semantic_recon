#!/usr/bin/env bash
# 03_tsdf_rgb.sh — Fuse RGB-D frames into a dense coloured mesh (TSDF).
# Edit the variables below, then run:  bash 03_tsdf_rgb.sh
set -e

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh
conda activate slam_recon

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── EDIT THESE ────────────────────────────────────────────────────────────────
FRAMES_DIR=/home/chengzhe/Data/OMS_data3/rs_bags/base_20260127_015119
TRAJECTORY=$PROJECT_DIR/output/base/sparse/trajectory_open3d.log
OUTPUT=$PROJECT_DIR/output/base/raw_mesh_rgb.ply

VOXEL_SIZE=0.005  # metres; 0.005 = 5 mm
DEPTH_MAX=3.0     # metres
DEPTH_MIN=0.15    # metres
CONFIDENCE=0      # 0=off, 1=drop zero-confidence, 2=keep high only
# ──────────────────────────────────────────────────────────────────────────────

python "$PROJECT_DIR/scripts/03_tsdf_rgb.py" \
    --frames_dir            "$FRAMES_DIR"  \
    --trajectory            "$TRAJECTORY"  \
    --output                "$OUTPUT"      \
    --voxel_size            "$VOXEL_SIZE"  \
    --depth_max             "$DEPTH_MAX"   \
    --depth_min             "$DEPTH_MIN"   \
    --confidence_threshold  "$CONFIDENCE"
