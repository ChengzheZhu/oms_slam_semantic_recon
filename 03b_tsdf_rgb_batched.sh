#!/usr/bin/env bash
# 03b_tsdf_rgb_batched.sh — Batched TSDF meshing; batches kept separate (no merge).
# Each batch of BATCH_SIZE (+BATCH_OVERLAP) frames uses its own fresh TSDF volume.
# Saves batch_NNNN.ply + index.json to OUTPUT_DIR/batches/.
# Use the same BATCH_SIZE / BATCH_OVERLAP in 05b to keep 1-to-1 correspondence.
set -e

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh
conda activate slam_recon

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── EDIT THESE ────────────────────────────────────────────────────────────────
FRAMES_DIR=/home/chengzhe/Data/OMS_data3/rs_bags/base_20260127_015119
TRAJECTORY=$PROJECT_DIR/output/base/sparse/trajectory_open3d.log
OUTPUT_DIR=$PROJECT_DIR/output/base-highres

VOXEL_SIZE=0.002    # metres
DEPTH_MAX=3.0
DEPTH_MIN=0.15
CONFIDENCE=0

BATCH_SIZE=100      # frames per batch — must match 05b
BATCH_OVERLAP=15    # overlap frames — must match 05b
# ──────────────────────────────────────────────────────────────────────────────

python "$PROJECT_DIR/scripts/03b_tsdf_rgb_batched.py" \
    --frames_dir            "$FRAMES_DIR"   \
    --trajectory            "$TRAJECTORY"   \
    --output_dir            "$OUTPUT_DIR"   \
    --voxel_size            "$VOXEL_SIZE"   \
    --depth_max             "$DEPTH_MAX"    \
    --depth_min             "$DEPTH_MIN"    \
    --confidence_threshold  "$CONFIDENCE"   \
    --batch_size            "$BATCH_SIZE"   \
    --batch_overlap         "$BATCH_OVERLAP"
