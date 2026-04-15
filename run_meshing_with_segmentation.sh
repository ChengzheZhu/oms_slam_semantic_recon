#!/usr/bin/env bash
# Re-run TSDF meshing + SAM3 segmentation on already-extracted frames and
# an already-computed ORB-SLAM3 trajectory. Skips bag extraction and SLAM.
#
# Usage:
#   ./run_meshing_with_segmentation.sh <slam_output_dir>
#
#   slam_output_dir  — existing output directory from a previous full-pipeline run.
#                      Must contain:
#                        sparse/trajectory_open3d.log
#                        config.env  (written automatically by run_*_pipeline.sh)
#
# A new timestamped directory is created as a sibling of slam_output_dir:
#   output/<dataset_name>_mesh_<timestamp>/
# SLAM results are never touched.

set -e

# ── config ────────────────────────────────────────────────────────────────────
CONDA_ENV="slam_recon"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAM3_PATH="$PROJECT_DIR/external/sam3"

SLAM_DIR="${1:-}"

if [ -z "$SLAM_DIR" ]; then
    echo "Usage: $0 <slam_output_dir>"
    echo ""
    echo "  slam_output_dir  existing output directory from a previous pipeline run"
    echo "                   (must contain sparse/trajectory_open3d.log and config.env)"
    exit 1
fi

SLAM_DIR="$(realpath "$SLAM_DIR")"
TRAJECTORY="$SLAM_DIR/sparse/trajectory_open3d.log"
CONFIG="$SLAM_DIR/config.env"

if [ ! -f "$TRAJECTORY" ]; then
    echo "ERROR: trajectory not found at $TRAJECTORY"
    echo "Run the full pipeline first to generate a trajectory."
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config.env not found at $CONFIG"
    echo "Re-run the full pipeline (it will write config.env automatically)."
    exit 1
fi

# Recover FRAMES_DIR saved by the pipeline script
source "$CONFIG"
FRAMES_DIR="$(realpath "$FRAMES_DIR")"

# New output dir — timestamped sibling so SLAM results are never overwritten
SLAM_BASENAME="$(basename "$SLAM_DIR")"
OUTPUT_DIR="$(dirname "$SLAM_DIR")/${SLAM_BASENAME}_mesh_$(date +%Y%m%d_%H%M%S)"

# Tunable parameters — mirror run_full_pipeline.sh defaults
FRAME_SUBSAMPLE=5
VOXEL_SIZE=0.005
SAM3_PROMPT="individual stone"
SAM3_CONFIDENCE=0.1
SAM3_MAX_SIZE_RATIO=0.15
ALPHA_THRESHOLD=0.1       # alpha score below which a vertex is seam/background
BOUNDARY_PROPAGATION_HOPS=0  # BFS hops to fatten seam network (0 = disabled)
MESH_KEEP_COMPONENTS=3        # drop all but the N largest TSDF components before segmentation
MIN_CLUSTER_SIZE=500

# Log
mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/meshing.log"
exec > >(tee -a "$LOG") 2>&1
echo "Log: $LOG"

# ── conda ─────────────────────────────────────────────────────────────────────
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export PYTHONPATH="$SAM3_PATH:$PYTHONPATH"

# ── run ───────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════"
echo " Meshing + Segmentation (no SLAM)"
echo " frames    : $FRAMES_DIR"
echo " trajectory: $TRAJECTORY"
echo " output    : $OUTPUT_DIR"
echo " voxel     : $VOXEL_SIZE m   subsample: $FRAME_SUBSAMPLE"
echo " seam hops : $BOUNDARY_PROPAGATION_HOPS   keep_components: $MESH_KEEP_COMPONENTS"
echo "════════════════════════════════════════"

# Step 1: SAM3 alpha cache (resumable — skips already-cached frames)
echo ""
echo "[1/3] Generating SAM3 alpha cache …"
python -u "$PROJECT_DIR/scripts/01b_run_sam3_alpha.py" \
    --frames_dir     "$FRAMES_DIR"      \
    --sam_prompt     "$SAM3_PROMPT"     \
    --sam_confidence "$SAM3_CONFIDENCE"

# Step 2: TSDF meshing + segmentation (SAM3 reads from cache)
echo ""
echo "[2/3] TSDF meshing + segmentation …"
python -u "$PROJECT_DIR/scripts/03d_sam3_boundary_reconstruction.py" \
    --frames_dir             "$FRAMES_DIR"                           \
    --intrinsic              "$FRAMES_DIR/intrinsic.json"            \
    --trajectory             "$TRAJECTORY"                           \
    --output                 "$OUTPUT_DIR/raw_mesh_rgb.ply"    \
    --segments_dir           "$OUTPUT_DIR/sam3_segments"             \
    --sam_prompt             "$SAM3_PROMPT"                          \
    --sam_confidence         "$SAM3_CONFIDENCE"                      \
    --sam_max_size_ratio     "$SAM3_MAX_SIZE_RATIO"                  \
    --frame_subsample        "$FRAME_SUBSAMPLE"                      \
    --voxel_size             "$VOXEL_SIZE"                           \
    --alpha_threshold        "$ALPHA_THRESHOLD"                      \
    --boundary_propagation_hops "$BOUNDARY_PROPAGATION_HOPS"         \
    --mesh_keep_components   "$MESH_KEEP_COMPONENTS"                 \
    --min_cluster_size       "$MIN_CLUSTER_SIZE"

# Step 3: Debug GLB
echo ""
echo "[3/3] Exporting debug GLB …"
python -u "$PROJECT_DIR/scripts/export_debug_glb.py" \
    --mesh   "$OUTPUT_DIR/raw_mesh_rgb.ply"     \
    --traj   "$TRAJECTORY"                            \
    --output "$OUTPUT_DIR/debug.glb"

echo ""
echo "════════════════════════════════════════"
echo " Done"
echo " mesh     : $OUTPUT_DIR/raw_mesh_rgb.ply"
echo " segments : $OUTPUT_DIR/sam3_segments/"
echo " debug    : $OUTPUT_DIR/debug.glb"
echo " log      : $LOG"
echo "════════════════════════════════════════"
