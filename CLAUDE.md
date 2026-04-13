# ORB_SLAM3 RGB-D Dense Reconstruction — Claude Code Context

## Project overview
Sub-project 2 of 3 in the OMS stone wall reconstruction pipeline:
1. **OMS components** (`~/Projects/oms`) — MASt3R/VGGT + SAM3 → per-stone point clouds
2. **This repo** — ORB-SLAM3 RGB-D SLAM + SAM3 boundary segmentation → assembly mesh
3. **Instance matching** (future) — geometry-based registration of individual stones into the assembly

Uses a RealSense D456 camera. ORB-SLAM3 tracks the camera; Open3D TSDF fuses depth into a
dense mesh; SAM3 boundary voting segments individual stone patches from the mesh.

## Repo layout
```
ORB_SLAM3_RGBD_DenseSlamReconstrction/
  config/
    camera/RealSense_D456.yaml   — camera intrinsics for ORB-SLAM3
  scripts/
    00_extract_frames.py         — .bag → PNG frames + intrinsic.json + streams.json
    01_run_orbslam3.sh           — runs ORB-SLAM3; passes viewer flag (0|1) as argv[5]
    02_convert_trajectory.py     — TUM format → Open3D log + pose graph JSON
    03_dense_reconstruction.py   — TSDF mesh (geometry only, no SAM3)
    03d_sam3_boundary_reconstruction.py — TSDF + SAM3 boundary voting → segmented mesh
    04_export_point_clouds.py    — export per-segment PLYs
    export_debug_glb.py          — mesh + camera frustums → GLB for geometry inspection
    create_associations.py       — TUM-format RGB-D associations file
    trim_bag.py                  — trim RealSense bag to a time window
  install/
    install_dependencies.sh      — system apt packages (Eigen, Boost, OpenCV, etc.)
    install_pangolin.sh          — builds Pangolin from source
    build_orbslam3.sh            — builds ORB-SLAM3 C++ from source
    setup_env.sh                 — creates slam_recon conda env + clones/installs SAM3
  external/
    orbslam3/                    — git submodule: ChengzheZhu/ORB_SLAM3.git
    sam3/                        — git submodule: ChengzheZhu/sam3.git (separate from OMS)
  environment.yml                — conda env spec (slam_recon, Python 3.11)
  run_full_pipeline.sh           — production: .bag → frames → SLAM → SAM3 mesh
  run_quick_test_pipeline.sh     — quick test: coarser params, same flow
```

## Environment
- Conda env: **`slam_recon`** (Python 3.11, PyTorch 2.7+cu126, Open3D 0.19)
- Create with: `bash install/setup_env.sh` (handles env + SAM3 editable install)
- SAM3 lives at `external/sam3` — **separate** from OMS project's `deps/sam3`
- Key version pins:
  - `opencv-python<4.10` — numpy<2 compatibility (SAM3 requires numpy<2)
  - `setuptools<71` — setuptools≥72 drops `pkg_resources` as top-level module
  - `psutil` — SAM3 transitive dep (eagerly imported in sam3_video_predictor.py)

## ORB-SLAM3 binary
- Built at `external/orbslam3/Examples/RGB-D/rgbd_tum`
- Viewer is a **runtime flag** — 5th argv `0` = headless, `1` = Pangolin viewer
- No recompile needed to switch modes; `01_run_orbslam3.sh --headless` passes `0`
- Vocabulary: `external/orbslam3/Vocabulary/ORBvoc.txt.tar.gz` → extract before first run

## Launching
```bash
# Activate env
conda activate slam_recon

# Full pipeline (bag → frames → ORB-SLAM3 → SAM3 mesh → debug GLB)
bash run_full_pipeline.sh /path/to/recording.bag

# Quick test (stride=3, coarser TSDF)
bash run_quick_test_pipeline.sh /path/to/recording.bag

# Resume from existing frames (skip re-extraction)
bash run_quick_test_pipeline.sh /path/to/recording.bag /path/to/existing/frames_dir

# Step by step
python scripts/00_extract_frames.py --bag recording.bag --output frames/ --stride 1
bash scripts/01_run_orbslam3.sh frames/ output/sparse/ --fps 30 --headless
python scripts/02_convert_trajectory.py \
    --input output/sparse/CameraTrajectory.txt \
    --output_log output/sparse/trajectory_open3d.log \
    --output_json output/sparse/trajectory_pose_graph.json
python scripts/03d_sam3_boundary_reconstruction.py \
    --frames_dir frames/ --intrinsic frames/intrinsic.json \
    --trajectory output/sparse/trajectory_open3d.log \
    --output output/mesh.ply --segments_dir output/segments/
```

## Key design decisions
- **ORB-SLAM3 submodule**: `external/orbslam3` — built in-place; vocabulary not in git
- **Runtime viewer toggle**: `rgbd_tum.cc` patched to accept argv[5] (0=headless, 1=viewer)
- **FRAME_STRIDE matters**: stride=3 (100ms/frame) can cause tracking loss on fast camera
  motion → prefer stride=1 (30fps, 33ms) for reliable single-map tracking
- **SAM3 boundary voting**: seam pixels = adjacent pixels with different mask IDs in SAM3
  output; boundary_vote_ratio threshold controls sensitivity
- **Depth filter**: always-on zero/min-range filter; optional confidence threshold for
  future bags that include a confidence stream
- **Frames stored outside repo**: extracted frames go beside the bag file

## On a new machine
```bash
git clone --recurse-submodules https://github.com/ChengzheZhu/ORB_SLAM3_RGBD_DenseSlamReconstrction.git
cd ORB_SLAM3_RGBD_DenseSlamReconstrction
sudo bash install/install_dependencies.sh
bash install/install_pangolin.sh
bash install/build_orbslam3.sh
cd external/orbslam3/Vocabulary && tar -xf ORBvoc.txt.tar.gz && cd -
bash install/setup_env.sh          # creates slam_recon env + installs SAM3
conda activate slam_recon
bash run_quick_test_pipeline.sh /path/to/recording.bag
```

## Bag files
- RealSense D456, RGB-D only (no confidence stream in current recordings)
- Location on server: `~/cloud/cheng-3dcv/OMS/2/rs_bags/`
- Bags are large (2–8 GB) on a network mount — run locally for stride=1 tests

## Data layout
- Output written to `output/<dataset_name>/` (gitignored):
  - `sparse/CameraTrajectory.txt` — ORB-SLAM3 TUM poses
  - `sparse/trajectory_open3d.log` — Open3D camera log
  - `sam3_boundary_mesh.ply` — TSDF mesh with SAM3 boundary segmentation
  - `sam3_segments/segment_*.ply` — per-stone patch PLYs
  - `debug.glb` — mesh + camera frustums for geometry inspection

## Pending tasks
- [ ] Spatial size filter on segments after mesh segmentation
- [ ] Wire `04_export_point_clouds.py` into `run_full_pipeline.sh`
- [ ] Server config: FRAME_STRIDE=3, VOXEL_SIZE=0.002 (wall motion is slower)
