# ORB_SLAM3 RGB-D Dense Reconstruction — Claude Code Context

## Project overview
RGB-D SLAM + dense 3D reconstruction pipeline for a dry-stacked stone wall (assembly).
Uses a RealSense D456 camera to capture RGB-D `.bag` files, runs ORB-SLAM3 for
camera tracking, then fuses depth frames into a dense TSDF mesh via Open3D.
Optional SAM3 segmentation identifies individual stone boundaries in the mesh.

## Repo layout
```
ORB_SLAM3_RGBD_DenseSlamReconstrction/
  bin/
    run_pipeline.py           — master entry point (YAML-config driven)
  config/
    camera/RealSense_D456.yaml — camera intrinsics for ORB-SLAM3
    pipeline/default.yaml      — pipeline config (set bag_file / frames_dir here)
    pipeline/examples/         — example configs for specific recordings
  scripts/
    00_extract_frames.py       — .bag → PNG frames + intrinsic.json
    01_run_orbslam3.sh         — runs ORB-SLAM3 executable
    02_convert_trajectory.py   — ORB-SLAM3 TUM format → Open3D log format
    03_dense_reconstruction.py — TSDF mesh from trajectory + depth frames
    03b–03e_*.py               — segmented / SAM3 variants
    04_export_point_clouds.py  — export individual/merged point clouds
    trim_bag.py                — trim RealSense bag to a time window
    create_associations.py     — create TUM-format RGB-D associations file
  install/
    install_dependencies.sh    — system apt packages (Eigen, Boost, etc.)
    build_orbslam3.sh          — builds ORB-SLAM3 C++ from source
  external/
    orbslam3/                  — git submodule: ChengzheZhu/ORB_SLAM3.git
  docs/                        — guides: setup, bag trimming, usage, etc.
  run_full_pipeline.sh         — one-shot: .bag → trajectory → SAM3 mesh
  run_quick_test_pipeline.sh   — same but every-5th-frame SAM3 subsample
  environment_sam3_open3d.yml  — conda env spec (Python 3.13, PyTorch, Open3D)
  requirements.txt             — pip dependencies
```

## Environment
- Conda env spec: `environment_sam3_open3d.yml`
  - Python 3.13, PyTorch 2.7+cu126, Open3D ≥0.18, pyrealsense2
- SAM3 must be installed editable from the local sam3 project:
  `pip install -e /path/to/sam3`
- ORB-SLAM3 built from source: `bash install/build_orbslam3.sh`
  - Requires system packages: `bash install/install_dependencies.sh`
  - Pangolin must be installed first: `bash scripts/install_pangolin.sh`
  - Vocabulary: `external/orbslam3/Vocabulary/ORBvoc.txt.tar.gz` → extract before first run

## Launching
```bash
# Full pipeline (bag → frames → ORB-SLAM3 → SAM3 mesh)
bash run_full_pipeline.sh /path/to/recording.bag

# Quick test (subsample every 5th frame for SAM3)
bash run_quick_test_pipeline.sh /path/to/recording.bag

# Config-driven (recommended for repeatable runs)
python bin/run_pipeline.py --config config/pipeline/default.yaml --bag /path/to/recording.bag

# Step by step
python scripts/00_extract_frames.py --bag recording.bag --output frames/ --stride 1
bash scripts/01_run_orbslam3.sh frames/ output/sparse/
python scripts/02_convert_trajectory.py --input output/sparse/CameraTrajectory.txt \
    --output output/sparse/trajectory_open3d.log
python scripts/03_dense_reconstruction.py --frames frames/ \
    --trajectory output/sparse/trajectory_open3d.log --output output/mesh.ply
```

## Key design decisions
- **ORB-SLAM3 as submodule**: `external/orbslam3` tracks ChengzheZhu/ORB_SLAM3.git;
  built in-place, not pip-installed
- **Vocabulary not in git**: `ORBvoc.txt` is ~600 MB; only the `.tar.gz` is tracked;
  extract it before first run
- **Frames stored outside repo**: extracted frames go to the same directory as the
  bag file to avoid polluting the repo (paths in config/default.yaml)
- **SAM3 boundary mode** (`03d_sam3_boundary_reconstruction.py`): identifies stone
  seams via SAM3 mask boundaries → better segmentation than raw TSDF alone
- **Headless mode**: set `use_viewer: false` in config for server runs (no display)

## On a cloud server
1. `git clone --recurse-submodules https://github.com/ChengzheZhu/ORB_SLAM3_RGBD_DenseSlamReconstrction.git`
2. `bash install/install_dependencies.sh`
3. `bash scripts/install_pangolin.sh`
4. `bash install/build_orbslam3.sh`
5. `cd external/orbslam3/Vocabulary && tar -xf ORBvoc.txt.tar.gz`
6. `conda env create -f environment_sam3_open3d.yml && conda activate sam3_open3d`
7. `pip install -e /path/to/sam3`
8. Set `use_viewer: false` in `config/pipeline/default.yaml` (no display on server)
9. `bash run_full_pipeline.sh /path/to/recording.bag`

## Data
- Input: RealSense D456 `.bag` files (RGB-D, stored externally)
- Output: written to `output/<dataset_name>/` (gitignored)
  - `sparse/CameraTrajectory.txt` — ORB-SLAM3 TUM-format poses
  - `sparse/trajectory_open3d.log` — Open3D camera log
  - `dense/mesh.ply` — TSDF fused mesh
  - `sam3_boundary_mesh.ply` — SAM3-segmented boundary mesh
