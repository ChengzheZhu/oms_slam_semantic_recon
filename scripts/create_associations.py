#!/usr/bin/env python3
"""
Create TUM-format association file for ORB_SLAM3.

Association format:
timestamp1 rgb/filename.jpg timestamp2 depth/filename.png
"""

import os
import argparse
from pathlib import Path


def create_associations(dataset_dir, output_file, fps=30.0):
    """
    Create TUM association file from extracted RealSense frames.

    Args:
        fps: Effective frame rate of the extracted frames. If you extracted
             with --stride N from a 30fps bag, pass fps=30/N so the FPS
             limiter in rgbd_tum.cc paces frames at the correct interval
             and gives the local-mapping / loop-closing threads enough time.
    """
    color_dir = os.path.join(dataset_dir, 'color')
    depth_dir = os.path.join(dataset_dir, 'depth')

    # Check for real timestamps saved during extraction
    timestamps_file = os.path.join(dataset_dir, 'timestamps.txt')
    real_timestamps = None
    if os.path.exists(timestamps_file):
        with open(timestamps_file) as f:
            real_timestamps = [float(line.strip()) for line in f if line.strip()]
        print(f"Using real bag timestamps from {timestamps_file}")

    # Get sorted lists of frames
    color_files = sorted([f for f in os.listdir(color_dir)
                         if f.endswith('.jpg') or f.endswith('.png')])
    depth_files = sorted([f for f in os.listdir(depth_dir)
                         if f.endswith('.png')])

    print(f"Found {len(color_files)} color frames")
    print(f"Found {len(depth_files)} depth frames")

    if len(color_files) != len(depth_files):
        print(f"WARNING: Mismatch in frame counts!")

    n_frames = min(len(color_files), len(depth_files))
    print(f"Creating associations for {n_frames} frames at {fps:.1f} fps")

    with open(output_file, 'w') as f:
        for i in range(n_frames):
            if real_timestamps and i < len(real_timestamps):
                timestamp = real_timestamps[i]
            else:
                timestamp = float(i) / fps

            f.write(f"{timestamp:.6f} color/{color_files[i]} ")
            f.write(f"{timestamp:.6f} depth/{depth_files[i]}\n")

    print(f"\n✓ Association file created: {output_file}")
    print(f"  Frames: {n_frames}")
    print(f"  Duration: {n_frames/fps:.2f} seconds (at {fps:.1f} fps)")


def main():
    parser = argparse.ArgumentParser(description="Create TUM association file")
    parser.add_argument('--dataset', type=str,
                        help='Dataset directory containing color/ and depth/ folders')
    parser.add_argument('--output', type=str,
                        help='Output association file')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Effective FPS of extracted frames (default 30). '
                             'Use 30/stride if frames were subsampled (e.g. 10 for stride=3). '
                             'Ignored if timestamps.txt exists in the dataset dir.')

    args = parser.parse_args()

    print("="*80)
    print("Creating TUM Association File")
    print("="*80)
    print(f"\nDataset: {args.dataset}")
    print(f"Output: {args.output}")
    print()

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    create_associations(args.dataset, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
