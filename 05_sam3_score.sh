#!/usr/bin/env bash
# 05_sam3_score.sh — EDT alpha scoring + semantic TSDF (L2 pass).
# Reads L1 mask cache from step 04 (set MASK_CACHE_DIR to match 04_sam3_mask.sh).
# Outputs: alpha_maps/ (L2 cache) + alpha_mesh.ply for step 06.
# Tuning edt_gamma or sam_max_size_ratio only re-runs EDT (fast), not SAM3.
# Edit the variables below, then run:  bash 05_sam3_score.sh
set -e

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh
conda activate slam_recon

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PROJECT_DIR/external/sam3:$PYTHONPATH"

# ── EDIT THESE ────────────────────────────────────────────────────────────────
FRAMES_DIR=/home/chengzhe/Data/OMS_data3/rs_bags/base_20260127_015119
TRAJECTORY=$PROJECT_DIR/output/base/sparse/trajectory_open3d.log
OUTPUT_DIR=$PROJECT_DIR/output/base/scoring

EDT_GAMMA=0.4          # <1 = sharp seams, =1 linear, >1 conservative
SAM_MAX_SIZE_RATIO=0.15

# L1 mask cache from step 04 — must match the cache dir produced by 04_sam3_mask.sh
MASK_CACHE_DIR=$FRAMES_DIR/sam3_mask_cache_conf_0.3

VOXEL_SIZE=0.005
DEPTH_MAX=3.0
DEPTH_MIN=0.15
# ──────────────────────────────────────────────────────────────────────────────

python "$PROJECT_DIR/scripts/05_sam3_score.py" \
    --frames_dir          "$FRAMES_DIR"          \
    --mask_cache_dir      "$MASK_CACHE_DIR"      \
    --trajectory          "$TRAJECTORY"          \
    --output_dir          "$OUTPUT_DIR"          \
    --edt_gamma           "$EDT_GAMMA"           \
    --sam_max_size_ratio  "$SAM_MAX_SIZE_RATIO"  \
    --voxel_size          "$VOXEL_SIZE"          \
    --depth_max           "$DEPTH_MAX"           \
    --depth_min           "$DEPTH_MIN"
