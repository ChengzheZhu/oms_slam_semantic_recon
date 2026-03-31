#!/bin/bash
set -e

echo "Installing Pangolin..."

# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    libglew-dev \
    libboost-dev \
    libboost-thread-dev \
    libboost-filesystem-dev \
    ffmpeg \
    libavcodec-dev \
    libavutil-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-dev \
    libraw1394-dev

# Clone Pangolin
PANGOLIN_DIR="/tmp/Pangolin"
if [ -d "$PANGOLIN_DIR" ]; then
    rm -rf "$PANGOLIN_DIR"
fi

git clone --recursive https://github.com/stevenlovegrove/Pangolin.git "$PANGOLIN_DIR"
cd "$PANGOLIN_DIR"

# Build and install
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install

# Cleanup
cd /tmp
rm -rf "$PANGOLIN_DIR"

echo "✓ Pangolin installed successfully"
