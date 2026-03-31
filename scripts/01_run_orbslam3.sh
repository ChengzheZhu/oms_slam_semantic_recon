#!/bin/bash
set -e

# ORB_SLAM3 RGB-D Pipeline for RealSense .bag files
# This script runs ORB_SLAM3 on a RealSense bag file and exports trajectory

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Configuration
ORB_SLAM3_DIR="$PROJECT_DIR/external/orbslam3"
VOCAB_FILE="$ORB_SLAM3_DIR/Vocabulary/ORBvoc.txt"
CONFIG_FILE="$PROJECT_DIR/config/camera/RealSense_D456.yaml"

# Parse arguments
HEADLESS_MODE=""
DATASET_DIR=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --headless)
            HEADLESS_MODE="no_viewer"
            shift
            ;;
        *)
            if [ -z "$DATASET_DIR" ]; then
                DATASET_DIR="$1"
            elif [ -z "$OUTPUT_DIR" ]; then
                OUTPUT_DIR="$1"
            fi
            shift
            ;;
    esac
done

# Set default dataset if not provided
if [ -z "$DATASET_DIR" ]; then
    DATASET_DIR="/home/chengzhe/Data/OMS_data3/rs_bags/1101/20251101_235516"
fi

# Set default output directory if not provided
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$PROJECT_DIR/output/sparse"
fi

# Convert OUTPUT_DIR to absolute path if it's relative
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$PROJECT_DIR/$OUTPUT_DIR"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "================================================"
echo "ORB_SLAM3 RGB-D SLAM Pipeline"
echo "================================================"

# Check vocabulary file
if [ ! -f "$VOCAB_FILE" ]; then
    echo -e "${RED}ERROR: Vocabulary file not found at $VOCAB_FILE${NC}"
    echo "Download from: https://github.com/UZ-SLAMLab/ORB_SLAM3/releases"
    exit 1
fi

# Check config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}ERROR: Config file not found at $CONFIG_FILE${NC}"
    exit 1
fi

# Check associations file
ASSOCIATIONS_FILE="$DATASET_DIR/associations.txt"
if [ ! -f "$ASSOCIATIONS_FILE" ]; then
    echo -e "${YELLOW}WARNING: associations.txt not found at $DATASET_DIR${NC}"
    echo "Generating associations file..."
    python3 "$SCRIPT_DIR/create_associations.py" --dataset "$DATASET_DIR" --output "$ASSOCIATIONS_FILE"
fi

# Check ORB_SLAM3 executable
# Current Pipeline uses frames. To use live sensor inputs, re-congifgure the pipe line to use "$ORB_SLAM3_DIR/Examples/RGB-D/rgbd_realsense_D435i"
ORBSLAM_EXEC="$ORB_SLAM3_DIR/Examples/RGB-D/rgbd_tum"
if [ ! -f "$ORBSLAM_EXEC" ]; then
    echo -e "${RED}ERROR: ORB_SLAM3 executable not found at $ORBSLAM_EXEC${NC}"
    echo "Please build ORB_SLAM3 first:"
    echo "  cd $ORB_SLAM3_DIR"
    echo "  ./build.sh"
    exit 1
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  ORB_SLAM3 Dir: $ORB_SLAM3_DIR"
echo "  Dataset: $DATASET_DIR"
echo "  Config: $CONFIG_FILE"
echo "  Output: $OUTPUT_DIR"
echo "  Mode: $([ -n "$HEADLESS_MODE" ] && echo "Headless" || echo "With Viewer")"

# Run ORB_SLAM3
echo -e "${YELLOW}Running ORB_SLAM3...${NC}"
cd "$ORB_SLAM3_DIR"

# Set library path for Pangolin
export LD_LIBRARY_PATH=$ORB_SLAM3_DIR/lib:/usr/local/lib:$LD_LIBRARY_PATH

# Run with or without viewer
if [ -n "$HEADLESS_MODE" ]; then
    echo "Running in headless mode (no visualization)"
    # Set environment variable to disable Pangolin viewer
    export DISPLAY=""
    ./Examples/RGB-D/rgbd_tum \
        "$VOCAB_FILE" \
        "$CONFIG_FILE" \
        "$DATASET_DIR" \
        "$ASSOCIATIONS_FILE" || echo "Note: ORB_SLAM3 may crash during cleanup (this is normal)"
else
    echo "Running with visualization enabled"
    ./Examples/RGB-D/rgbd_tum \
        "$VOCAB_FILE" \
        "$CONFIG_FILE" \
        "$DATASET_DIR" \
        "$ASSOCIATIONS_FILE" || echo "Note: ORB_SLAM3 may crash during cleanup (this is normal)"
fi

# Check if trajectory files were generated (what matters)
if [ ! -f "CameraTrajectory.txt" ]; then
    echo -e "${RED}ERROR: ORB_SLAM3 failed - no trajectory file generated${NC}"
    exit 1
fi

echo -e "${GREEN}✓ ORB_SLAM3 tracking completed successfully${NC}"

# Copy trajectory output to output directory
if [ -f "CameraTrajectory.txt" ]; then
    cp CameraTrajectory.txt "$OUTPUT_DIR/CameraTrajectory.txt"
    echo -e "${GREEN}Trajectory saved to: $OUTPUT_DIR/CameraTrajectory.txt${NC}"
fi

if [ -f "KeyFrameTrajectory.txt" ]; then
    cp KeyFrameTrajectory.txt "$OUTPUT_DIR/KeyFrameTrajectory.txt"
    echo -e "${GREEN}KeyFrame trajectory saved to: $OUTPUT_DIR/KeyFrameTrajectory.txt${NC}"
fi

echo -e "${GREEN}ORB_SLAM3 Complete!${NC}"
echo "================================================"
