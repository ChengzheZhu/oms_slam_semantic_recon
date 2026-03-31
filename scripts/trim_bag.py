#!/usr/bin/env python3
"""
Interactive RealSense .bag file trimmer.

Allows you to select a specific portion of a .bag file to extract
the best reconstruction segment.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os
import json
from pathlib import Path


def get_bag_info(bag_file):
    """Get information about the bag file."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)

    print(f"\nAnalyzing bag file: {bag_file}")
    profile = pipeline.start(config)

    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    # Get duration
    duration = playback.get_duration().total_seconds()

    # Count frames
    frame_count = 0
    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            frame_count += 1
    except RuntimeError:
        pass

    pipeline.stop()

    return {
        'frame_count': frame_count,
        'duration': duration,
        'fps': frame_count / duration if duration > 0 else 0
    }


def preview_frames(bag_file, start_frame=0, end_frame=None, skip=1):
    """Preview frames in the bag file with interactive controls."""

    # First pass: load all frames into memory for slider support
    print(f"\nLoading frames for preview...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)

    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    frames_data = []
    frame_idx = 0

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()

            if color_frame and frame_idx >= start_frame and (end_frame is None or frame_idx <= end_frame):
                if (frame_idx - start_frame) % skip == 0:
                    color_image = np.asanyarray(color_frame.get_data())
                    frames_data.append((frame_idx, color_image.copy()))

            frame_idx += 1

            if len(frames_data) % 100 == 0 and len(frames_data) > 0:
                print(f"  Loaded {len(frames_data)} frames...")

            if end_frame is not None and frame_idx > end_frame:
                break

    except RuntimeError:
        pass

    pipeline.stop()

    if not frames_data:
        print("ERROR: No frames loaded")
        return []

    print(f"✓ Loaded {len(frames_data)} frames")
    print(f"\nPreview Controls:")
    print("  SLIDER - Jump to frame")
    print("  SPACE - Play/Pause")
    print("  RIGHT ARROW - Next frame")
    print("  LEFT ARROW - Previous frame")
    print("  's' - Save frame marker (start/end)")
    print("  'c' - Clear all markers")
    print("  'q' - Quit and use markers")

    # Create window
    window_name = 'Bag Preview - Use Slider & Keys'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # State
    current_idx = 0
    playing = False
    saved_frames = []

    def on_trackbar(val):
        nonlocal current_idx
        current_idx = val

    # Create trackbar
    cv2.createTrackbar('Frame', window_name, 0, len(frames_data) - 1, on_trackbar)

    def draw_frame():
        """Draw current frame with overlays."""
        frame_num, color_image = frames_data[current_idx]
        display = color_image.copy()

        # Add frame info
        info_text = f"Frame: {frame_num} / {frames_data[-1][0]}"
        cv2.putText(display, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Add status
        status = "PLAYING" if playing else "PAUSED"
        color = (0, 255, 0) if playing else (0, 165, 255)
        cv2.putText(display, status, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Add markers
        if saved_frames:
            markers_text = f"Markers: {len(saved_frames)} - {sorted(saved_frames)}"
            cv2.putText(display, markers_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Highlight if current frame is a marker
            if frame_num in saved_frames:
                cv2.rectangle(display, (0, 0), (display.shape[1], display.shape[0]),
                            (0, 255, 255), 10)
                cv2.putText(display, "MARKER", (display.shape[1]//2 - 80, display.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

        cv2.imshow(window_name, display)

    # Main loop
    while True:
        # Update trackbar position
        cv2.setTrackbarPos('Frame', window_name, current_idx)

        # Draw current frame
        draw_frame()

        # Auto-advance if playing
        if playing:
            current_idx = min(current_idx + 1, len(frames_data) - 1)
            if current_idx >= len(frames_data) - 1:
                playing = False

        # Handle key presses
        wait_time = 30 if playing else 1
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            playing = not playing
            print(f"  {'Playing' if playing else 'Paused'}")
        elif key == ord('s'):
            frame_num = frames_data[current_idx][0]
            if frame_num in saved_frames:
                saved_frames.remove(frame_num)
                print(f"  Removed marker: {frame_num}")
            else:
                saved_frames.append(frame_num)
                print(f"  Saved marker: {frame_num}")
            saved_frames.sort()
        elif key == ord('c'):
            saved_frames.clear()
            print(f"  Cleared all markers")
        elif key == 83:  # Right arrow
            current_idx = min(current_idx + 1, len(frames_data) - 1)
        elif key == 81:  # Left arrow
            current_idx = max(current_idx - 1, 0)
        elif key == 82:  # Up arrow - skip forward
            current_idx = min(current_idx + 10, len(frames_data) - 1)
        elif key == 84:  # Down arrow - skip backward
            current_idx = max(current_idx - 10, 0)

    cv2.destroyAllWindows()

    return saved_frames


def trim_bag_to_frames(input_bag, output_dir, start_frame, end_frame):
    """Extract trimmed frames directly (more reliable than bag creation)."""
    print(f"\nExtracting trimmed frames...")
    print(f"  Input bag: {input_bag}")
    print(f"  Output dir: {output_dir}")
    print(f"  Frames: {start_frame} to {end_frame}")

    # Create output directories
    color_dir = os.path.join(output_dir, 'color')
    depth_dir = os.path.join(output_dir, 'depth')
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # Open bag file
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(input_bag, repeat_playback=False)

    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    # Get camera intrinsics
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        print("ERROR: Could not get initial frames")
        pipeline.stop()
        return 0

    # Get intrinsics
    color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # Save intrinsics to JSON
    intrinsic_data = {
        "width": color_intrinsics.width,
        "height": color_intrinsics.height,
        "intrinsic_matrix": [
            color_intrinsics.fx, 0.0, 0.0,
            0.0, color_intrinsics.fy, 0.0,
            color_intrinsics.ppx, color_intrinsics.ppy, 1.0
        ],
        "depth_scale": 1.0 / depth_scale
    }

    intrinsic_file = os.path.join(output_dir, 'intrinsic.json')
    with open(intrinsic_file, 'w') as f:
        json.dump(intrinsic_data, f, indent=2)
    print(f"✓ Saved intrinsics: {intrinsic_file}")

    # Extract frames
    frame_idx = 0
    saved_count = 0

    print(f"\nExtracting frames {start_frame} to {end_frame}...")

    try:
        while frame_idx <= end_frame:
            frames = pipeline.wait_for_frames(timeout_ms=1000)

            if frame_idx >= start_frame:
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if color_frame and depth_frame:
                    # Save color
                    color_image = np.asanyarray(color_frame.get_data())
                    color_path = os.path.join(color_dir, f"{saved_count:06d}.png")
                    cv2.imwrite(color_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

                    # Save depth
                    depth_image = np.asanyarray(depth_frame.get_data())
                    depth_path = os.path.join(depth_dir, f"{saved_count:06d}.png")
                    cv2.imwrite(depth_path, depth_image)

                    saved_count += 1

                    if saved_count % 100 == 0:
                        print(f"  Extracted {saved_count} frames...")

            frame_idx += 1

    except RuntimeError:
        pass

    pipeline.stop()

    # Create associations file
    print(f"\nCreating associations file...")
    assoc_file = os.path.join(output_dir, 'associations.txt')
    with open(assoc_file, 'w') as f:
        for i in range(saved_count):
            timestamp = i / 30.0  # Assuming 30 FPS
            f.write(f"{timestamp:.6f} color/{i:06d}.png {timestamp:.6f} depth/{i:06d}.png\n")

    print(f"✓ Created {assoc_file}")
    print(f"\n✓ Extraction complete!")
    print(f"  Frames saved: {saved_count}")
    print(f"  Output: {output_dir}")

    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Interactive RealSense bag file trimmer"
    )
    parser.add_argument('--bag', type=str, required=True,
                       help='Input .bag file')
    parser.add_argument('--output', type=str,
                       help='Output .bag file (auto-generated if not specified)')
    parser.add_argument('--start', type=int,
                       help='Start frame (will prompt if not specified)')
    parser.add_argument('--end', type=int,
                       help='End frame (will prompt if not specified)')
    parser.add_argument('--preview', action='store_true',
                       help='Preview frames before trimming')
    parser.add_argument('--skip', type=int, default=1,
                       help='Frame skip for loading (default: 1, use higher for memory savings)')

    args = parser.parse_args()

    if not os.path.exists(args.bag):
        print(f"ERROR: Bag file not found: {args.bag}")
        return 1

    print("="*80)
    print("RealSense Bag File Trimmer")
    print("="*80)

    # Get bag info
    info = get_bag_info(args.bag)
    print(f"\nBag Info:")
    print(f"  Total frames: {info['frame_count']}")
    print(f"  Duration: {info['duration']:.2f} seconds")
    print(f"  FPS: {info['fps']:.2f}")

    # Interactive mode if start/end not specified
    if args.start is None or args.end is None:
        print("\n" + "="*80)
        print("INTERACTIVE MODE")
        print("="*80)
        print("\nOptions:")
        print("  1. Preview and select frames")
        print("  2. Enter frame range manually")

        choice = input("\nChoice (1/2): ").strip()

        if choice == '1' or args.preview:
            print("\nPreviewing frames...")
            print("Controls:")
            print("  - Use SLIDER to jump to any frame")
            print("  - Press SPACE to play/pause")
            print("  - Press ARROW KEYS to step frame-by-frame")
            print("  - Press 's' to mark start/end frames")
            print("  - Press 'c' to clear all markers")
            print("  - Press 'q' to finish and use markers")

            saved = preview_frames(args.bag, 0, None, args.skip)

            if len(saved) >= 2:
                print(f"\nSaved frame markers: {saved}")
                start = saved[0]
                end = saved[-1]
                confirm = input(f"Use frames {start} to {end}? (y/n): ")
                if confirm.lower() == 'y':
                    args.start = start
                    args.end = end

        if args.start is None:
            args.start = int(input(f"\nStart frame (0-{info['frame_count']-1}): "))

        if args.end is None:
            args.end = int(input(f"End frame ({args.start}-{info['frame_count']-1}): "))

    # Validate range
    if args.start < 0 or args.end >= info['frame_count']:
        print(f"ERROR: Invalid frame range")
        return 1

    if args.start >= args.end:
        print(f"ERROR: Start frame must be less than end frame")
        return 1

    # Generate output directory if not specified
    if args.output is None:
        input_path = Path(args.bag)
        args.output = str(input_path.parent / f"{input_path.stem}_trimmed_{args.start}_{args.end}")

    # Confirm
    num_frames = args.end - args.start + 1
    duration = num_frames / info['fps']
    print(f"\nTrim Summary:")
    print(f"  Frames: {args.start} to {args.end} ({num_frames} frames)")
    print(f"  Duration: ~{duration:.2f} seconds")
    print(f"  Output: {args.output}/")

    confirm = input("\nProceed with extraction? (y/n): ")
    if confirm.lower() != 'y':
        print("Cancelled")
        return 0

    # Extract frames
    num_extracted = trim_bag_to_frames(args.bag, args.output, args.start, args.end)

    if num_extracted == 0:
        print("ERROR: No frames extracted")
        return 1

    print("\n" + "="*80)
    print("Extraction Complete!")
    print("="*80)
    print(f"\nExtracted {num_extracted} frames to:")
    print(f"  {args.output}/")
    print(f"\nFiles created:")
    print(f"  - color/ ({num_extracted} frames)")
    print(f"  - depth/ ({num_extracted} frames)")
    print(f"  - intrinsic.json")
    print(f"  - associations.txt")
    print(f"\nNext step: Run pipeline on extracted frames:")
    print(f"  cd /home/chengzhe/projects/_OMS/ORB_SLAM3_RGBD_DenseSlamReconstrction")
    print(f"  # Skip step 00 (frames already extracted)")
    print(f"  bash scripts/01_run_orbslam3.sh \"{args.output}\" output/trimmed_run/sparse")
    print(f"  # Then continue with steps 02 and 03d...")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
