#!/bin/bash
# Build ORB_SLAM3 from the git submodule at external/orbslam3/
#
# Prerequisites (run once before this script):
#   bash install/install_dependencies.sh
#   bash install/install_pangolin.sh
#
# Usage:
#   bash install/build_orbslam3.sh [--no-viewer]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ORBSLAM3_DIR="$PROJECT_DIR/external/orbslam3"

# ── 1. Ensure submodule is populated ─────────────────────────────────────────
if [ ! -f "$ORBSLAM3_DIR/CMakeLists.txt" ]; then
    echo "Initialising git submodule…"
    cd "$PROJECT_DIR"
    git submodule update --init --recursive external/orbslam3
fi

# ── 2. Build (headless — no Pangolin viewer needed) ───────────────────────────
# We always run headless (server or local test via GLB export).
# Pangolin may still be installed on the system; this just disables the viewer.
CMAKE_FLAGS="-DENABLE_PANGOLIN=0"

echo "Building ORB_SLAM3 at: $ORBSLAM3_DIR (headless, no viewer)"
cd "$ORBSLAM3_DIR"

# Run the bundled build script (builds DBoW2, g2o, Sophus, then the main lib
# and RGB-D examples in one go)
chmod +x build.sh
./build.sh $CMAKE_FLAGS

# ── 4. Extract vocabulary ─────────────────────────────────────────────────────
VOC="$ORBSLAM3_DIR/Vocabulary/ORBvoc.txt"
VOC_TGZ="$ORBSLAM3_DIR/Vocabulary/ORBvoc.txt.tar.gz"
if [ ! -f "$VOC" ]; then
    echo "Extracting vocabulary…"
    tar -xf "$VOC_TGZ" -C "$ORBSLAM3_DIR/Vocabulary/"
fi

echo ""
echo "✓ ORB_SLAM3 build complete"
echo "  Binary: $ORBSLAM3_DIR/Examples/RGB-D/rgbd_tum"
echo "  Vocab : $VOC"
