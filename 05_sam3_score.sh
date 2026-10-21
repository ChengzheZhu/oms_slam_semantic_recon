#!/usr/bin/env bash
# 05_sam3_score.sh — EDT alpha scoring + semantic TSDF (L2 pass).
# Reads the L1 mask cache from step 04 (dir derived from sam3.confidence +
# sam3.skip_qr_recovery). Outputs: alpha_maps/ (L2 cache) + alpha_mesh.ply for step 06.
# Tuning score.edt_gamma or score.sam_max_size_ratio only re-runs EDT (fast), not SAM3.
# Settings: pipeline.yaml (paths.*, tsdf.*, score.*). Run:  bash 05_sam3_score.sh
set -e

# ── OVERRIDES — this run only; uncomment to replace pipeline.yaml values ──────
# MASK_CACHE_DIR=/abs/path/to/sam3_mask_cache_conf_0.3_qr_filled
# SCORING_DIR=/abs/path/to/scoring
# SCORE_EDT_GAMMA=0.6
# ──────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/scripts/pipeline_env.sh"
export PYTHONPATH="$PROJECT_DIR/external/sam3:$PYTHONPATH"

python "$PROJECT_DIR/scripts/05_sam3_score.py" \
    --frames_dir          "$FRAMES_DIR"               \
    --mask_cache_dir      "$MASK_CACHE_DIR"           \
    --trajectory          "$TRAJECTORY"               \
    --output_dir          "$SCORING_DIR"              \
    --edt_gamma           "$SCORE_EDT_GAMMA"          \
    --sam_max_size_ratio  "$SCORE_SAM_MAX_SIZE_RATIO" \
    --voxel_size          "$TSDF_VOXEL_SIZE"          \
    --depth_max           "$TSDF_DEPTH_MAX"           \
    --depth_min           "$TSDF_DEPTH_MIN"
