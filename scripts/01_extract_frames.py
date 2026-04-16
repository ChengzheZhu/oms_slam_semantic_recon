#!/usr/bin/env python3
"""
01_extract_frames.py — Extract RGB-D frames from a RealSense .bag file.

Outputs:
  <output_dir>/color/NNNNNN.jpg      — colour frames
  <output_dir>/depth/NNNNNN.png      — depth frames (uint16)
  <output_dir>/confidence/NNNNNN.png — confidence frames (if present in bag)
  <output_dir>/intrinsic.json        — camera intrinsics (Open3D column-major)
  <output_dir>/timestamps.txt        — per-frame bag timestamps (seconds)
  <output_dir>/streams.json          — stream availability metadata

Usage:
  python scripts/01_extract_frames.py --bag /path/to/recording.bag --output /path/to/frames_dir
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
import argparse


def extract_frames_from_bag(bag_file, output_dir, frame_stride=1, max_frames=0,
                            skip_confidence=False):
    color_dir = os.path.join(output_dir, 'color')
    depth_dir = os.path.join(output_dir, 'depth')
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)

    print(f"Opening bag file: {bag_file}")
    profile  = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    frames      = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        print("ERROR: Could not get initial frames")
        pipeline.stop()
        return

    has_confidence = (not skip_confidence) and bool(
        frames.first_or_default(rs.stream.confidence))
    conf_dir = os.path.join(output_dir, 'confidence')
    if has_confidence:
        os.makedirs(conf_dir, exist_ok=True)
        print("  Confidence stream: FOUND — extracting to confidence/")
    else:
        print(f"  Confidence stream: not present{' (skipped)' if skip_confidence else ''}")

    ci = color_frame.profile.as_video_stream_profile().intrinsics
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    print(f"\nCamera: {ci.width}x{ci.height}  "
          f"fx={ci.fx:.2f} fy={ci.fy:.2f}  "
          f"cx={ci.ppx:.2f} cy={ci.ppy:.2f}  depth_scale={depth_scale}")

    intrinsic_data = {
        "width":  ci.width,
        "height": ci.height,
        "intrinsic_matrix": [ci.fx, 0.0, 0.0, 0.0, ci.fy, 0.0, ci.ppx, ci.ppy, 1.0],
        "depth_scale": 1.0 / depth_scale,
    }
    with open(os.path.join(output_dir, 'intrinsic.json'), 'w') as f:
        json.dump(intrinsic_data, f, indent=2)

    frame_count = saved_count = 0
    timestamps  = []

    print(f"\nExtracting frames (stride={frame_stride})...")
    try:
        while True:
            if max_frames > 0 and saved_count >= max_frames:
                print(f"Reached max_frames limit: {max_frames}")
                break
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                break

            if frame_count % frame_stride != 0:
                frame_count += 1
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                frame_count += 1
                continue

            color_np = np.asanyarray(color_frame.get_data())
            depth_np = np.asanyarray(depth_frame.get_data())
            ts       = frames.get_timestamp() / 1000.0

            cv2.imwrite(os.path.join(color_dir, f'{saved_count:06d}.jpg'),
                        cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(depth_dir, f'{saved_count:06d}.png'), depth_np)

            if has_confidence:
                cf = frames.first_or_default(rs.stream.confidence)
                if cf:
                    cv2.imwrite(os.path.join(conf_dir, f'{saved_count:06d}.png'),
                                np.asanyarray(cf.get_data()))

            timestamps.append(ts)
            saved_count  += 1
            frame_count  += 1

            if saved_count % 100 == 0:
                print(f"  {saved_count} frames extracted…")

    except Exception as e:
        print(f"Error during extraction: {e}")
    finally:
        pipeline.stop()

    with open(os.path.join(output_dir, 'timestamps.txt'), 'w') as f:
        for ts in timestamps:
            f.write(f"{ts:.6f}\n")

    with open(os.path.join(output_dir, 'streams.json'), 'w') as f:
        json.dump({"has_confidence": has_confidence,
                   "frame_count": saved_count,
                   "frame_stride": frame_stride}, f, indent=2)

    print(f"\n✓ Extraction complete: {saved_count} frames → {output_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract RGB-D frames from a RealSense .bag file")
    parser.add_argument('--bag',    required=True, help='.bag file path')
    parser.add_argument('--output', required=True, help='Output frames directory')
    parser.add_argument('--stride', type=int, default=1,
                        help='Extract every Nth frame (default: 1 = all)')
    parser.add_argument('--max_frames', type=int, default=0,
                        help='Max frames to extract (0 = all)')
    parser.add_argument('--skip_confidence', action='store_true',
                        help='Skip confidence stream even if present')
    args = parser.parse_args()

    if not os.path.exists(args.bag):
        print(f"ERROR: bag not found: {args.bag}")
        return 1

    print("=" * 60)
    print("Step 01 — Extract Frames")
    print("=" * 60)
    extract_frames_from_bag(args.bag, args.output, args.stride,
                            args.max_frames, args.skip_confidence)


if __name__ == "__main__":
    main()
