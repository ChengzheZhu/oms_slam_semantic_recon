#!/usr/bin/env python3
"""
01_extract_frames.py — Extract RGB-D frames from a RealSense .bag file.

Outputs:
  <output_dir>/color/NNNNNN.jpg      — colour frames
  <output_dir>/depth/NNNNNN.png      — depth frames (uint16), aligned to colour
  <output_dir>/confidence/NNNNNN.png — confidence frames (if present in bag)
  <output_dir>/intrinsic.json        — colour intrinsics (Open3D column-major);
                                       valid for depth too since depth is aligned
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


def align_confidence_to_color(conf_np, aligned_depth_np, depth_units, ci, di, d2c):
    """Resample a depth-frame confidence image into the colour frame.

    rs.align only moves the depth stream, so confidence would no longer line up
    with the aligned depth. For every colour pixel with valid aligned depth,
    deproject with the colour K, move into the depth camera, project with the
    depth K and take the nearest confidence value. Pixels with no depth get 0.
    """
    h, w = aligned_depth_np.shape
    v, u = np.mgrid[0:h, 0:w]
    z = aligned_depth_np.astype(np.float32) * depth_units
    valid = z > 0
    pts_c = np.stack([(u - ci.ppx) / ci.fx * z,
                      (v - ci.ppy) / ci.fy * z, z], axis=-1)[valid]
    # librealsense extrinsics: column-major R, p_color = R @ p_depth + t
    R = np.array(d2c.rotation, dtype=np.float32).reshape(3, 3).T
    t = np.array(d2c.translation, dtype=np.float32)
    pts_d = (pts_c - t) @ R              # R^T (p_c - t), row-vector form
    zd = pts_d[:, 2]
    ud = np.round(pts_d[:, 0] / zd * di.fx + di.ppx).astype(np.int64)
    vd = np.round(pts_d[:, 1] / zd * di.fy + di.ppy).astype(np.int64)
    inb = (zd > 0) & (ud >= 0) & (ud < conf_np.shape[1]) & (vd >= 0) & (vd < conf_np.shape[0])
    vals = np.zeros(len(zd), dtype=conf_np.dtype)
    vals[inb] = conf_np[vd[inb], ud[inb]]
    out = np.zeros((h, w), dtype=conf_np.dtype)
    out[valid] = vals
    return out


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

    # Bag stores raw depth in the depth-camera frame (~59 mm from colour on
    # D456, different K). Reproject it into the colour frame so depth pixels
    # line up with colour pixels and the colour K applies to both.
    align = rs.align(rs.stream.color)

    frames      = align.process(pipeline.wait_for_frames())
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
    di = profile.get_stream(rs.stream.depth).as_video_stream_profile().intrinsics
    d2c = profile.get_stream(rs.stream.depth).get_extrinsics_to(
        profile.get_stream(rs.stream.color))
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    print(f"\nCamera: {ci.width}x{ci.height}  "
          f"fx={ci.fx:.2f} fy={ci.fy:.2f}  "
          f"cx={ci.ppx:.2f} cy={ci.ppy:.2f}  depth_scale={depth_scale}")

    intrinsic_data = {
        "width":  ci.width,
        "height": ci.height,
        "intrinsic_matrix": [ci.fx, 0.0, 0.0, 0.0, ci.fy, 0.0, ci.ppx, ci.ppy, 1.0],
        "depth_scale": 1.0 / depth_scale,
        "depth_aligned_to_color": True,
        # Metadata only — load_intrinsic() ignores these keys.
        "color_distortion": {"model": str(ci.model), "coeffs": list(ci.coeffs)},
        "raw_depth_intrinsic": {"width": di.width, "height": di.height,
                                "fx": di.fx, "fy": di.fy,
                                "cx": di.ppx, "cy": di.ppy},
        "depth_to_color": {"rotation": list(d2c.rotation),
                           "translation": list(d2c.translation)},
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

            frames      = align.process(frames)
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
                    conf_np = align_confidence_to_color(
                        np.asanyarray(cf.get_data()), depth_np, depth_scale,
                        ci, di, d2c)
                    cv2.imwrite(os.path.join(conf_dir, f'{saved_count:06d}.png'),
                                conf_np)

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
