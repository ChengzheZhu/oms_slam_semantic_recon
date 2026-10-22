#!/usr/bin/env bash
# 05_sam3_score.sh — Seam votes + semantic TSDF (L2 pass).
# Reads step 04's merged mask cache (dir derived from sam3.prompt, sam3.label_prompt,
# sam3.confidence, sam3.skip_label_recovery). Outputs: edt_maps/ (L2 cache) + vote_mesh.ply for step 06.
# Tuning score.seam_width_mm / seam_gamma only re-runs the TSDF; sam_max_size_ratio also the EDT.
# Settings: pipeline.yaml (paths.*, tsdf.*, score.*). Run:  bash 05_sam3_score.sh
set -e

# ── OVERRIDES — this run only; uncomment to replace pipeline.yaml values ──────
# MASK_CACHE_DIR=/abs/path/to/sam3_merged_individual_stone+paper_label_conf_0.5
# SCORING_DIR=/abs/path/to/scoring
# SCORE_SEAM_WIDTH_MM=8
# ──────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/scripts/pipeline_env.sh"
export PYTHONPATH="$PROJECT_DIR/external/sam3:$PYTHONPATH"

python "$PROJECT_DIR/scripts/05_sam3_score.py" \
    --frames_dir          "$FRAMES_DIR"               \
    --mask_cache_dir      "$MASK_CACHE_DIR"           \
    --trajectory          "$TRAJECTORY"               \
    --output_dir          "$SCORING_DIR"              \
    --seam_width_mm       "$SCORE_SEAM_WIDTH_MM"      \
    --seam_gamma          "$SCORE_SEAM_GAMMA"         \
    --sam_max_size_ratio  "$SCORE_SAM_MAX_SIZE_RATIO" \
    --voxel_size          "$TSDF_VOXEL_SIZE"          \
    --depth_max           "$TSDF_DEPTH_MAX"           \
    --depth_min           "$TSDF_DEPTH_MIN"
