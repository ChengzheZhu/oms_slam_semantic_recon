#!/usr/bin/env python3
"""
Extract RGB-D frames from RealSense .bag file.

This script:
1. Reads a RealSense .bag file
2. Extracts RGB and depth frames
3. Saves camera intrinsics to JSON
4. Outputs frames to color/ and depth/ directories
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
import argparse
from pathlib import Path


def extract_frames_from_bag(bag_file, output_dir, frame_stride=1, max_frames=0,
                            skip_confidence=False):
    """
    Extract RGB-D frames from RealSense bag file.

    Args:
        bag_file: Path to .bag file
        output_dir: Directory to save extracted frames
        frame_stride: Extract every Nth frame (1 = all frames)
        max_frames: Maximum number of frames to extract (0 = all)
        skip_confidence: Skip confidence stream even if present in bag
    """
    # Create output directories
    color_dir = os.path.join(output_dir, 'color')
    depth_dir = os.path.join(output_dir, 'depth')
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)

    # Start pipeline
    print(f"Opening bag file: {bag_file}")
    profile = pipeline.start(config)

    # Get device and sensor info
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)  # Process as fast as possible

    # Probe for confidence stream on first frameset
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        print("ERROR: Could not get initial frames")
        pipeline.stop()
        return

    has_confidence = (not skip_confidence) and bool(frames.first_or_default(rs.stream.confidence))
    conf_dir = os.path.join(output_dir, 'confidence')
    if has_confidence:
        os.makedirs(conf_dir, exist_ok=True)
        print(f"  Confidence stream: FOUND — extracting to confidence/")
    else:
        print(f"  Confidence stream: not present{' (skipped)' if skip_confidence else ''}")

    # Get intrinsics
    color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    print(f"\nCamera Info:")
    print(f"  Resolution: {color_intrinsics.width}x{color_intrinsics.height}")
    print(f"  Focal: fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}")
    print(f"  Principal: cx={color_intrinsics.ppx:.2f}, cy={color_intrinsics.ppy:.2f}")
    print(f"  Depth scale: {depth_scale}")

    # Save intrinsics to JSON
    intrinsic_data = {
        "width": color_intrinsics.width,
        "height": color_intrinsics.height,
        "intrinsic_matrix": [
            color_intrinsics.fx, 0.0, 0.0,
            0.0, color_intrinsics.fy, 0.0,
            color_intrinsics.ppx, color_intrinsics.ppy, 1.0
        ],
        "depth_scale": 1.0 / depth_scale  # Open3D format
    }

    intrinsic_file = os.path.join(output_dir, 'intrinsic.json')
    with open(intrinsic_file, 'w') as f:
        json.dump(intrinsic_data, f, indent=2)
    print(f"✓ Saved intrinsics: {intrinsic_file}")

    # Extract frames
    frame_count = 0
    saved_count = 0
    timestamps = []

    print(f"\nExtracting frames (stride={frame_stride})...")

    try:
        while True:
            # Check max frames limit
            if max_frames > 0 and saved_count >= max_frames:
                print(f"Reached max frames limit: {max_frames}")
                break

            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                # End of bag file
                break

            # Apply frame stride
            if frame_count % frame_stride != 0:
                frame_count += 1
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                frame_count += 1
                continue

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Capture real bag timestamp (ms → seconds) before saving
            frame_timestamp_sec = frames.get_timestamp() / 1000.0

            # Save frames
            color_filename = os.path.join(color_dir, f'{saved_count:06d}.jpg')
            depth_filename = os.path.join(depth_dir, f'{saved_count:06d}.png')

            cv2.imwrite(color_filename, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(depth_filename, depth_image)

            # Save confidence frame if available
            if has_confidence:
                conf_frame = frames.first_or_default(rs.stream.confidence)
                if conf_frame:
                    conf_image = np.asanyarray(conf_frame.get_data())
                    cv2.imwrite(os.path.join(conf_dir, f'{saved_count:06d}.png'), conf_image)

            timestamps.append(frame_timestamp_sec)
            saved_count += 1
            frame_count += 1

            if saved_count % 100 == 0:
                print(f"  Extracted {saved_count} frames...")

    except Exception as e:
        print(f"Error during extraction: {e}")
    finally:
        pipeline.stop()

    # Write real bag timestamps so create_associations.py uses actual timing
    timestamps_file = os.path.join(output_dir, 'timestamps.txt')
    with open(timestamps_file, 'w') as f:
        for ts in timestamps:
            f.write(f"{ts:.6f}\n")

    print(f"\n✓ Extraction complete!")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Frames saved: {saved_count}")
    print(f"  Confidence stream: {'extracted' if has_confidence else 'not available'}")
    print(f"  Timestamps: {timestamps_file}")
    print(f"  Output: {output_dir}")

    # Write stream availability so downstream scripts can auto-detect
    streams_info = {
        "has_confidence": has_confidence,
        "frame_count": saved_count,
        "frame_stride": frame_stride,
    }
    with open(os.path.join(output_dir, 'streams.json'), 'w') as f:
        json.dump(streams_info, f, indent=2)

    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Extract frames from RealSense bag file")
    parser.add_argument('--bag', type=str, required=True,
                       help='Path to RealSense .bag file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for extracted frames')
    parser.add_argument('--stride', type=int, default=1,
                       help='Extract every Nth frame (default: 1 = all frames)')
    parser.add_argument('--max_frames', type=int, default=0,
                       help='Maximum number of frames to extract (0 = all)')
    parser.add_argument('--skip_confidence', action='store_true',
                       help='Skip confidence stream even if present in bag')

    args = parser.parse_args()

    print("="*80)
    print("RealSense Bag Frame Extraction")
    print("="*80)

    if not os.path.exists(args.bag):
        print(f"ERROR: Bag file not found: {args.bag}")
        return 1

    extract_frames_from_bag(args.bag, args.output, args.stride, args.max_frames,
                            skip_confidence=args.skip_confidence)

    print("\n" + "="*80)
    print("Next step: Run ORB_SLAM3 pipeline")
    print("  ./run_pipeline.py --config config/pipeline_config.yaml")
    print("="*80)


if __name__ == "__main__":
    main()
