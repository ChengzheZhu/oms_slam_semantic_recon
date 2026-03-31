#!/bin/bash
set -e

# Quick Test Pipeline with Frame Subsampling
# Faster version for testing - extracts every 10th frame

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BAG_FILE="${1:-${BAG_FILE:-}}"   # pass as first arg or set BAG_FILE env var
if [ -z "$BAG_FILE" ]; then
    echo "Usage: $0 /path/to/recording.bag"
    echo "  or:  BAG_FILE=/path/to/recording.bag $0"
    exit 1
fi
DATASET_NAME="$(basename "${BAG_FILE%.bag}")_sam3_$(date +%Y%m%d_%H%M%S)"
FRAMES_DIR="$(dirname "$BAG_FILE")/${DATASET_NAME}"
OUTPUT_DIR="${PROJECT_DIR}/output/${DATASET_NAME}"

echo "=============================================="
echo "SAM3 Pipeline (All Frames, Every 5th for SAM3)"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  BAG file: $BAG_FILE"
echo "  Frame stride: 1 (all frames for ORB-SLAM3)"
echo "  SAM3 subsample: 5 (every 5th frame for reconstruction)"
echo "  Frames output: $FRAMES_DIR"
echo "  Pipeline output: $OUTPUT_DIR"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

cd $PROJECT_DIR

# Step 0: Extract all frames
echo ""
echo -e "${GREEN}Step 0: Extracting all frames (stride=1)...${NC}"
python scripts/00_extract_frames.py \
  --bag "$BAG_FILE" \
  --output "$FRAMES_DIR" \
  --stride 1

echo -e "${GREEN}✓ Frames extracted${NC}"

# Step 1: Run ORB-SLAM3
echo ""
echo -e "${GREEN}Step 1: Running ORB-SLAM3 (with viewer for stability)...${NC}"
bash scripts/01_run_orbslam3.sh "$FRAMES_DIR" "$OUTPUT_DIR/sparse"

echo -e "${GREEN}✓ ORB-SLAM3 complete${NC}"

# Step 2: Convert trajectory
echo ""
echo -e "${GREEN}Step 2: Converting trajectory...${NC}"
python scripts/02_convert_trajectory.py \
  --input "$OUTPUT_DIR/sparse/CameraTrajectory.txt" \
  --output "$OUTPUT_DIR/sparse/trajectory_open3d.log"

echo -e "${GREEN}✓ Trajectory converted${NC}"

# Step 3: SAM3 boundary reconstruction
echo ""
echo -e "${GREEN}Step 3: SAM3 boundary reconstruction (subsample=5)...${NC}"
python scripts/03d_sam3_boundary_reconstruction.py \
  --frames_dir "$FRAMES_DIR" \
  --intrinsic "$FRAMES_DIR/intrinsic.json" \
  --trajectory "$OUTPUT_DIR/sparse/trajectory_open3d.log" \
  --output "$OUTPUT_DIR/sam3_boundary_mesh.ply" \
  --segments_dir "$OUTPUT_DIR/sam3_segments" \
  --sam_prompt "individual stone" \
  --sam_confidence 0.1 \
  --sam_max_size_ratio 0.15 \
  --frame_subsample 5 \
  --voxel_size 0.005 \
  --boundary_threshold 0.01 \
  --min_cluster_size 500

echo ""
echo "=============================================="
echo -e "${GREEN}Pipeline Complete!${NC}"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  Full mesh: $OUTPUT_DIR/sam3_boundary_mesh.ply"
echo "  Boundaries: $OUTPUT_DIR/sam3_boundary_mesh_boundaries.ply"
echo "  Segments: $OUTPUT_DIR/sam3_segments/"
echo "  Trajectory: $OUTPUT_DIR/sparse/trajectory_open3d.log"
echo ""
echo "To visualize:"
echo "  python -c \"import open3d as o3d; mesh = o3d.io.read_triangle_mesh('$OUTPUT_DIR/sam3_boundary_mesh.ply'); o3d.visualization.draw_geometries([mesh])\""
