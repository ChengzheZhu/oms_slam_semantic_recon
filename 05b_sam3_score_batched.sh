#!/usr/bin/env bash
# 05b_sam3_score_batched.sh — Batched semantic TSDF; vote batches kept separate.
# Uses the same batch.size / batch.overlap as 03b so each vote_batch_NNNN.ply
# corresponds 1-to-1 with batch_NNNN.ply from 03b.
# Saves vote_batch_NNNN.ply + index.json to <run dir>/scoring/vote_batches/.
# Settings: pipeline.yaml (paths.*, tsdf.*, batch.*, score.*). Run:  bash 05b_sam3_score_batched.sh
set -e

# ── OVERRIDES — this run only; uncomment to replace pipeline.yaml values ──────
# MASK_CACHE_DIR=/abs/path/to/sam3_merged_individual_stone+paper_label_conf_0.5
# SCORE_SEAM_WIDTH_MM=8
# BATCH_SIZE=50                     # keep equal in 03b
# ──────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/scripts/pipeline_env.sh"
export PYTHONPATH="$PROJECT_DIR/external/sam3:$PYTHONPATH"

python "$PROJECT_DIR/scripts/05b_sam3_score_batched.py" \
    --frames_dir          "$FRAMES_DIR"               \
    --mask_cache_dir      "$MASK_CACHE_DIR"           \
    --trajectory          "$TRAJECTORY"               \
    --output_dir          "$SCORING_DIR"              \
    --seam_width_mm       "$SCORE_SEAM_WIDTH_MM"      \
    --seam_gamma          "$SCORE_SEAM_GAMMA"         \
    --sam_max_size_ratio  "$SCORE_SAM_MAX_SIZE_RATIO" \
    --voxel_size          "$TSDF_VOXEL_SIZE"          \
    --depth_max           "$TSDF_DEPTH_MAX"           \
    --depth_min           "$TSDF_DEPTH_MIN"           \
    --batch_size          "$BATCH_SIZE"               \
    --batch_overlap       "$BATCH_OVERLAP"
