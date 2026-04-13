#!/usr/bin/env bash
# Quick test: same as run_full_pipeline.sh but with coarser frame sampling.
# Useful for verifying the pipeline end-to-end before a full run.
#
# Usage:
#   ./run_quick_test_pipeline.sh /path/to/recording.bag
#   BAG_FILE=/path/to/recording.bag ./run_quick_test_pipeline.sh
#
# Output: output/<dataset_name>_test/

set -e

# ── config ────────────────────────────────────────────────────────────────────
CONDA_ENV="slam_recon"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAM3_PATH="$PROJECT_DIR/external/sam3"

# Input — positional args or env vars
#   $1  path to .bag file
#   $2  (optional) pre-extracted frames dir — skips step 0 if valid
BAG_FILE="${1:-${BAG_FILE:-}}"
FRAMES_DIR_OVERRIDE="${2:-${FRAMES_DIR_OVERRIDE:-}}"

if [ -z "$BAG_FILE" ]; then
    echo "Usage: $0 /path/to/recording.bag [/path/to/existing/frames_dir]"
    exit 1
fi

# Tunable parameters (coarser than full pipeline for speed)
FRAME_STRIDE=3          # extract every Nth frame for ORB-SLAM3
SLAM_FPS=10             # effective FPS = 30 / FRAME_STRIDE (30/3=10)
                        # keeps rgbd_tum.cc sleep intervals correct
FRAME_SUBSAMPLE=10      # use every Nth frame for SAM3 reconstruction
VOXEL_SIZE=0.01         # coarser voxel for faster test
SAM3_PROMPT="individual stone"
SAM3_CONFIDENCE=0.1
SAM3_MAX_SIZE_RATIO=0.15
BOUNDARY_VOTE_RATIO=0.3
MIN_CLUSTER_SIZE=200
USE_VIEWER=false        # headless — binary compiled without viewer

# Derived paths — use override frames dir if supplied
if [ -n "$FRAMES_DIR_OVERRIDE" ]; then
    FRAMES_DIR="$(realpath "$FRAMES_DIR_OVERRIDE")"
    DATASET_NAME="$(basename "$FRAMES_DIR")"
else
    DATASET_NAME="$(basename "${BAG_FILE%.bag}")_test_$(date +%Y%m%d_%H%M%S)"
    FRAMES_DIR="$(dirname "$BAG_FILE")/${DATASET_NAME}"
fi
OUTPUT_DIR="${PROJECT_DIR}/output/${DATASET_NAME}"

# Log file (timestamped, kept in output dir)
mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Log: $LOG"

# ── conda ─────────────────────────────────────────────────────────────────────
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export PYTHONPATH="$SAM3_PATH:$PYTHONPATH"

# ── run ───────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════"
echo " ORB-SLAM3 + SAM3 — Quick Test"
echo " bag     : $BAG_FILE"
echo " frames  : $FRAMES_DIR"
echo " output  : $OUTPUT_DIR"
echo " env     : $CONDA_ENV"
echo " stride  : $FRAME_STRIDE  subsample: $FRAME_SUBSAMPLE"
echo "════════════════════════════════════════"

# Step 0: Extract frames (skip if valid extraction already exists)
echo ""
_frames_ok() {
    [ -d "$1/color" ] && [ "$(ls -A "$1/color" 2>/dev/null)" ] && [ -f "$1/intrinsic.json" ]
}
if _frames_ok "$FRAMES_DIR"; then
    _n=$(ls "$FRAMES_DIR/color" | wc -l)
    echo "[0/3] Skipping extraction — $FRAMES_DIR already has $_n colour frames"
else
    echo "[0/3] Extracting frames (stride=$FRAME_STRIDE) …"
    python -u "$PROJECT_DIR/scripts/00_extract_frames.py" \
        --bag    "$BAG_FILE"   \
        --output "$FRAMES_DIR" \
        --stride "$FRAME_STRIDE"
fi

# Step 1: ORB-SLAM3 tracking
echo ""
echo "[1/3] Running ORB-SLAM3 …"
VIEWER_ARG=""
[ "$USE_VIEWER" = "false" ] && VIEWER_ARG="--headless"
bash "$PROJECT_DIR/scripts/01_run_orbslam3.sh" "$FRAMES_DIR" "$OUTPUT_DIR/sparse" --fps "$SLAM_FPS" $VIEWER_ARG

mkdir -p "$OUTPUT_DIR/sparse"
cp "$PROJECT_DIR/external/orbslam3/CameraTrajectory.txt"   "$OUTPUT_DIR/sparse/"
cp "$PROJECT_DIR/external/orbslam3/KeyFrameTrajectory.txt"  "$OUTPUT_DIR/sparse/"

# Step 2: Convert trajectory
echo ""
echo "[2/3] Converting trajectory to Open3D format …"
python -u "$PROJECT_DIR/scripts/02_convert_trajectory.py" \
    --input      "$OUTPUT_DIR/sparse/CameraTrajectory.txt"        \
    --output_log "$OUTPUT_DIR/sparse/trajectory_open3d.log"       \
    --output_json "$OUTPUT_DIR/sparse/trajectory_pose_graph.json"

# Step 3: SAM3 boundary reconstruction
echo ""
echo "[3/3] SAM3 boundary reconstruction (subsample=$FRAME_SUBSAMPLE) …"
python -u "$PROJECT_DIR/scripts/03d_sam3_boundary_reconstruction.py" \
    --frames_dir        "$FRAMES_DIR"                              \
    --intrinsic         "$FRAMES_DIR/intrinsic.json"              \
    --trajectory        "$OUTPUT_DIR/sparse/trajectory_open3d.log" \
    --output            "$OUTPUT_DIR/sam3_boundary_mesh.ply"       \
    --segments_dir      "$OUTPUT_DIR/sam3_segments"                \
    --sam_prompt        "$SAM3_PROMPT"                             \
    --sam_confidence    "$SAM3_CONFIDENCE"                         \
    --sam_max_size_ratio "$SAM3_MAX_SIZE_RATIO"                    \
    --frame_subsample   "$FRAME_SUBSAMPLE"                         \
    --voxel_size        "$VOXEL_SIZE"                              \
    --boundary_vote_ratio "$BOUNDARY_VOTE_RATIO"                   \
    --min_cluster_size  "$MIN_CLUSTER_SIZE"

echo ""
echo "════════════════════════════════════════"
echo " Done"
echo " mesh     : $OUTPUT_DIR/sam3_boundary_mesh.ply"
echo " segments : $OUTPUT_DIR/sam3_segments/"
echo " log      : $LOG"
echo "════════════════════════════════════════"
