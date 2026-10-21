#!/usr/bin/env bash
# 02_slam.sh — Run ORB-SLAM3 RGB-D tracking and convert the trajectory.
# Saves the Atlas so other sensors can relocalize against this map.
# Settings: pipeline.yaml (paths.*, slam.*, orbslam.*). Run:  bash 02_slam.sh
# Writes sparse/orbslam_settings.yaml next to the atlas for relocalisation.
set -e

# ── OVERRIDES — this run only; uncomment to replace pipeline.yaml values ──────
# FRAMES_DIR=/abs/path/to/frames
# SPARSE_DIR=/abs/path/to/sparse
# SLAM_EQUALIZE=false
# SLAM_LOAD_ATLAS=/abs/path/to/atlas      # no .osa
# SLAM_LOCALIZE=true
# ORBSLAM_N_FEATURES=6000
# ──────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/scripts/pipeline_env.sh"

ATLAS_ARGS=()
[ "$SLAM_SAVE_ATLAS" = true ] && ATLAS_ARGS+=(--save_atlas "$SPARSE_DIR/atlas")
[ -n "$SLAM_LOAD_ATLAS" ]     && ATLAS_ARGS+=(--load_atlas "$SLAM_LOAD_ATLAS")

python "$PROJECT_DIR/scripts/02_slam.py" \
    --frames_dir "$FRAMES_DIR"         \
    --output_dir "$SPARSE_DIR"         \
    --vocab      "$SLAM_VOCAB"         \
    --fps        "$SLAM_FPS"           \
    --clahe_clip_limit "$SLAM_CLAHE_CLIP_LIMIT" \
    --clahe_tile_grid  "$SLAM_CLAHE_TILE_GRID"  \
    --orb_n_features   "$ORBSLAM_N_FEATURES"   \
    --orb_scale_factor "$ORBSLAM_SCALE_FACTOR" \
    --orb_n_levels     "$ORBSLAM_N_LEVELS"     \
    --orb_ini_th_fast  "$ORBSLAM_INI_TH_FAST"  \
    --orb_min_th_fast  "$ORBSLAM_MIN_TH_FAST"  \
    --th_depth         "$ORBSLAM_TH_DEPTH"     \
    --stereo_b         "$ORBSLAM_STEREO_B"     \
    "${ATLAS_ARGS[@]}"                 \
    $(flag "$SLAM_VIEWER"   --viewer)   \
    $(flag "$SLAM_EQUALIZE" --equalize) \
    $(flag "$SLAM_LOCALIZE" --localize)
