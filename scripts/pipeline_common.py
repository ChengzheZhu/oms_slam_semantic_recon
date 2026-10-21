#!/usr/bin/env python3
"""
pipeline_common.py — shared IO helpers for the reconstruction stages
(03/03b RGB TSDF, 05/05b SAM3 scoring).

Poses are matched to frames by timestamp (load_frame_poses), never by position:
ORB-SLAM3 leaves untracked frames out of the trajectory.

Extracted verbatim from the per-stage copies. `apply_depth_filter` keeps the
confidence-aware superset (from 03/03b); 05/05b call it with defaults, so their
behaviour is unchanged (confidence_np=None skips the confidence branch).
"""

import json
import os

import numpy as np
import open3d as o3d


def load_trajectory_log(log_file):
    """Read trajectory_open3d.log → (timestamps, poses).

    Current format (02_slam.py): `timestamp m00 … m33` per line, camera-to-world
    row-major. Legacy files have the 16 matrix values only; timestamps is then
    None and poses are in tracked order with lost frames missing.
    """
    stamps, poses, legacy = [], [], False
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) == 17:
                stamps.append(vals[0])
                poses.append(np.array(vals[1:]).reshape(4, 4))
            elif len(vals) == 16:
                legacy = True
                poses.append(np.array(vals).reshape(4, 4))
    return (None if legacy else stamps), poses


def load_frame_poses(log_file, frames_dir, max_dt=1e-3):
    """Pose for every frame in frames_dir, matched by timestamp.

    Frame i's time is line i of <frames_dir>/timestamps.txt (written by step 01
    alongside color/NNNNNN.jpg). Each frame takes the trajectory pose nearest in
    time, if within max_dt seconds (both sides are written to the microsecond;
    frames are >= 11 ms apart). Frames ORB-SLAM3 did not track get None —
    callers skip them instead of fusing a neighbour's pose.
    """
    color_files, depth_files = get_rgbd_file_lists(frames_dir)
    n_frames = min(len(color_files), len(depth_files))
    stamps, traj = load_trajectory_log(log_file)
    ts_file = os.path.join(frames_dir, 'timestamps.txt')

    if stamps is None or not os.path.exists(ts_file):
        why = ("legacy trajectory without timestamps" if stamps is None
               else f"no {ts_file}")
        if len(traj) != n_frames:
            raise SystemExit(
                f"ERROR: {why}, and {len(traj)} poses != {n_frames} frames — poses "
                f"cannot be matched to frames. Regenerate the trajectory with:\n"
                f"  python scripts/02_slam.py --convert_only --frames_dir {frames_dir} "
                f"--output_dir {os.path.dirname(os.path.abspath(log_file))}")
        print(f"  WARNING: {why}; pairing poses with frames by index "
              f"(counts match: {n_frames})")
        return list(traj)

    with open(ts_file) as f:
        frame_ts = [float(l) for l in f if l.strip()]
    order = np.argsort(stamps)
    traj_ts = np.asarray(stamps)[order]
    poses = [None] * n_frames
    for i, t in enumerate(frame_ts[:n_frames]):
        j = int(np.searchsorted(traj_ts, t))
        best = min((k for k in (j - 1, j) if 0 <= k < len(traj_ts)),
                   key=lambda k: abs(traj_ts[k] - t), default=None)
        if best is not None and abs(traj_ts[best] - t) <= max_dt:
            poses[i] = traj[order[best]]

    n_posed = sum(p is not None for p in poses)
    if n_posed == 0:
        raise SystemExit(f"ERROR: no trajectory timestamp in {log_file} matches "
                         f"{ts_file} — trajectory from a different extraction?")
    print(f"  Poses: {n_posed}/{n_frames} frames matched by timestamp"
          + (f" ({n_frames - n_posed} untracked frames skipped)"
             if n_posed < n_frames else ""))
    return poses


def load_intrinsic(intrinsic_file):
    with open(intrinsic_file) as f:
        d = json.load(f)
    m = d['intrinsic_matrix']
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=d['width'], height=d['height'],
        fx=m[0], fy=m[4], cx=m[6], cy=m[7])
    return intrinsic, d.get('depth_scale', 1000.0)


def get_rgbd_file_lists(frames_dir):
    color_dir = os.path.join(frames_dir, 'color')
    depth_dir = os.path.join(frames_dir, 'depth')
    color_files = sorted(os.path.join(color_dir, f) for f in os.listdir(color_dir)
                         if f.endswith(('.jpg', '.png')))
    depth_files = sorted(os.path.join(depth_dir, f) for f in os.listdir(depth_dir)
                         if f.endswith('.png'))
    return color_files, depth_files


def apply_depth_filter(depth_np, depth_scale, min_depth_m=0.15,
                       confidence_np=None, confidence_threshold=0):
    min_raw = int(min_depth_m * depth_scale)
    invalid = (depth_np == 0) | (depth_np < min_raw)
    if confidence_np is not None and confidence_threshold > 0:
        invalid |= (confidence_np < confidence_threshold)
    if invalid.any():
        depth_np = depth_np.copy()
        depth_np[invalid] = 0
    return depth_np
