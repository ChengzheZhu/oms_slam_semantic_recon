#!/usr/bin/env bash
# 03b_tsdf_rgb_batched.sh — Batched TSDF meshing; batches kept separate (no merge).
# Each batch of batch.size (+batch.overlap) frames uses its own fresh TSDF volume.
# Saves batch_NNNN.ply + index.json to <run dir>/batches/.
# Settings: pipeline.yaml (paths.*, tsdf.*, batch.*). Run:  bash 03b_tsdf_rgb_batched.sh
set -e

# ── OVERRIDES — this run only; uncomment to replace pipeline.yaml values ──────
# RUN_DIR=/abs/path/to/run          # batches/ is written under this
# TSDF_VOXEL_SIZE=0.005
# BATCH_SIZE=50                     # keep equal in 05b
# ──────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/scripts/pipeline_env.sh"

python "$PROJECT_DIR/scripts/03b_tsdf_rgb_batched.py" \
    --frames_dir            "$FRAMES_DIR"                \
    --trajectory            "$TRAJECTORY"                \
    --output_dir            "$RUN_DIR"                   \
    --voxel_size            "$TSDF_VOXEL_SIZE"           \
    --depth_max             "$TSDF_DEPTH_MAX"            \
    --depth_min             "$TSDF_DEPTH_MIN"            \
    --confidence_threshold  "$TSDF_CONFIDENCE_THRESHOLD" \
    --batch_size            "$BATCH_SIZE"                \
    --batch_overlap         "$BATCH_OVERLAP"
