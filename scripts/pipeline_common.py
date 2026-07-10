#!/usr/bin/env python3
"""
pipeline_common.py — shared IO helpers for the reconstruction stages
(03/03b RGB TSDF, 05/05b SAM3 scoring).

Extracted verbatim from the per-stage copies. `apply_depth_filter` keeps the
confidence-aware superset (from 03/03b); 05/05b call it with defaults, so their
behaviour is unchanged (confidence_np=None skips the confidence branch).
"""

import json
import os

import numpy as np
import open3d as o3d


def load_trajectory_log(log_file):
    poses = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) == 16:
                poses.append(np.array(vals).reshape(4, 4))
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
