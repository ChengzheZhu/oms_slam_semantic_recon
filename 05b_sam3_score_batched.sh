#!/usr/bin/env bash
# 05b_sam3_score_batched.sh — Batched semantic TSDF; alpha batches kept separate.
# Matches 03b batch structure (same BATCH_SIZE / BATCH_OVERLAP) so each
# alpha_batch_NNNN.ply corresponds 1-to-1 with batch_NNNN.ply from 03b.
# Saves alpha_batch_NNNN.ply + index.json to OUTPUT_DIR/alpha_batches/.
set -e

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh
conda activate slam_recon

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PROJECT_DIR/external/sam3:$PYTHONPATH"

# ── EDIT THESE ────────────────────────────────────────────────────────────────
FRAMES_DIR=/home/chengzhe/Data/OMS_data3/rs_bags/base_20260127_015119
TRAJECTORY=$PROJECT_DIR/output/base/sparse/trajectory_open3d.log
OUTPUT_DIR=$PROJECT_DIR/output/base-highres/scoring

EDT_GAMMA=0.4
SAM_MAX_SIZE_RATIO=0.15

MASK_CACHE_DIR=$FRAMES_DIR/sam3_mask_cache_conf_0.5_qr_filled

VOXEL_SIZE=0.002
DEPTH_MAX=3.0
DEPTH_MIN=0.15

# Must match 03b values for 1-to-1 batch correspondence
BATCH_SIZE=100
BATCH_OVERLAP=15
# ──────────────────────────────────────────────────────────────────────────────

python "$PROJECT_DIR/scripts/05b_sam3_score_batched.py" \
    --frames_dir          "$FRAMES_DIR"          \
    --mask_cache_dir      "$MASK_CACHE_DIR"      \
    --trajectory          "$TRAJECTORY"          \
    --output_dir          "$OUTPUT_DIR"          \
    --edt_gamma           "$EDT_GAMMA"           \
    --sam_max_size_ratio  "$SAM_MAX_SIZE_RATIO"  \
    --voxel_size          "$VOXEL_SIZE"          \
    --depth_max           "$DEPTH_MAX"           \
    --depth_min           "$DEPTH_MIN"           \
    --batch_size          "$BATCH_SIZE"          \
    --batch_overlap       "$BATCH_OVERLAP"
