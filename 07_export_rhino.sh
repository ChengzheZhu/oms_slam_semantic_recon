#!/usr/bin/env bash
# 07_export_rhino.sh — Export TSDF points (cloud.ply) and camera cones +
# trajectory (cameras.obj) for Rhino, into <run dir>/export/.
# Settings: pipeline.yaml (paths.*, export.*). Run:  bash 07_export_rhino.sh
set -e

# ── OVERRIDES — this run only; uncomment to replace pipeline.yaml values ──────
# EXPORT_DIR=/abs/path/to/export
# EXPORT_SOURCE=batches
# EXPORT_UNITS=mm
# EXPORT_VOXEL=0.005
# EXPORT_CONE_EVERY=30
# ──────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/scripts/pipeline_env.sh"

case "$EXPORT_SOURCE" in
    mesh)    MESHES=("$RAW_MESH") ;;
    batches) MESHES=("$RUN_DIR"/batches/batch_*.ply) ;;
    *) echo "export.source must be mesh or batches (got '$EXPORT_SOURCE')" >&2; exit 1 ;;
esac

python "$PROJECT_DIR/scripts/07_export_rhino.py" \
    --frames_dir  "$FRAMES_DIR"        \
    --trajectory  "$TRAJECTORY"        \
    --mesh        "${MESHES[@]}"       \
    --output_dir  "$EXPORT_DIR"        \
    --units       "$EXPORT_UNITS"      \
    --voxel       "$EXPORT_VOXEL"      \
    --cone_depth  "$EXPORT_CONE_DEPTH" \
    --cone_every  "$EXPORT_CONE_EVERY" \
    $(flag "$EXPORT_NORMALS" --normals)
