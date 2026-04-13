#!/bin/bash
# Install all dependencies for slam_dense_reconstruction

set -e

echo "Installing system dependencies..."

# Update package list
sudo apt-get update

# Install build essentials
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config

# Install Eigen3
sudo apt-get install -y libeigen3-dev

# RealSense SDK: installed via pip (pyrealsense2), not needed for building ORB-SLAM3
# librealsense2-dev / librealsense2-utils skipped

# Install GTK for OpenCV
sudo apt-get install -y \
    libgtk2.0-dev \
    libgtk-3-dev

# Install Pangolin dependencies
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

# Install Python dependencies
echo "Installing Python packages..."
pip install -r ../requirements.txt

echo "✓ All dependencies installed successfully!"
echo "Next: Run ./build_orbslam3.sh to build ORB_SLAM3"
