# OMS — SLAM Semantic Reconstruction (assembly)

RGB-D SLAM + semantic segmentation of a dry-stacked stone **wall (the assembly)**.
ORB-SLAM3 tracks the camera, Open3D fuses depth into a dense TSDF mesh, and SAM3
EDT alpha-scoring separates individual stone interiors from the seams between them.

Sub-project **2 of 3** in the OMS stone-wall pipeline:
1. **Components** (`oms_monocular_semantic_recon`) — per-stone reconstruction (MASt3R/VGGT + SAM3)
2. **Assembly** *(this repo)* — SLAM + semantic segmentation of the whole wall
3. **Registration** *(future)* — geometry-based matching of stones into the assembly

Input: a RealSense D456 `.bag`. Output: a segmented assembly mesh (per-stone submeshes + seams culled).

## Pipeline

Six stages. Each stage is a root wrapper script `NN_*.sh` (edit its config block) over
`scripts/NN_*.py`. `run_pipeline.sh` runs the non-batched track end to end.

| # | Run | Does |
|---|-----|------|
| 01 | `01_extract.sh` | `.bag` → `color/`, `depth/`, `confidence/`, `intrinsic.json`, `timestamps.txt`, `streams.json` |
| 02 | `02_slam.sh` | ORB-SLAM3 RGB-D tracking + trajectory conversion → `trajectory_open3d.log` (+ pose-graph JSON) |
| 03 | `03_tsdf_rgb.sh` | TSDF-fuse all frames → `raw_mesh_rgb.ply` (geometry) |
| 04 | `04_sam3_mask.sh` | SAM3 **L1** mask cache; stone + QR prompts (one shared image encode); QR-hole recovery |
| 05 | `05_sam3_score.sh` | **L2**: per-frame EDT alpha maps → semantic TSDF → `alpha_mesh.ply` |
| 06 | `06_cull_segment.sh` | transfer alpha scores to the RGB mesh, cull seam triangles, segment into stone submeshes |

**Two tracks:**
- **Non-batched** (`03 → 05 → 06`) — single TSDF volume; fits in memory. Orchestrated by `run_pipeline.sh`.
- **Batched** (`03b → 05b → 06b`) — for large / high-res (2 mm voxel) scenes: overlapping temporal
  batches kept as separate meshes. `batch_size` / `batch_overlap` **must match** between `03b` and `05b`.
  Stages `01/02/04` are shared. See [`docs/drafts/0421_dev_notes.md`](docs/drafts/0421_dev_notes.md) for the rationale.

## Quick start

```bash
conda activate slam_recon

# Full non-batched pipeline (edit dataset paths at the top of each NN_*.sh first)
bash run_pipeline.sh              # all stages 01–06
bash run_pipeline.sh 3 6          # stages 03 through 06 only

# Or run a single stage
bash 04_sam3_mask.sh
```

## Installation

See [`docs/SETUP.md`](docs/SETUP.md) for the full guide. In short:

```bash
git clone --recurse-submodules https://github.com/ChengzheZhu/oms_slam_semantic_recon.git
cd oms_slam_semantic_recon

sudo bash install/install_dependencies.sh   # system libs
bash install/install_pangolin.sh            # Pangolin (ORB-SLAM3 viewer)
bash install/build_orbslam3.sh              # build ORB-SLAM3 → Examples/RGB-D/rgbd_tum
cd external/orbslam3/Vocabulary && tar -xf ORBvoc.txt.tar.gz && cd -   # ORB vocabulary
bash install/setup_env.sh                   # create slam_recon env + install SAM3 (external/sam3)
```

- **Env:** `slam_recon` (Python 3.11, PyTorch 2.7 + cu126, Open3D 0.19) — spec in `environment.yml`.
- **Submodules:** `external/orbslam3` (ORB-SLAM3 fork), `external/sam3` (SAM3 fork — separate from the components repo).
- **ORB-SLAM3 viewer** is a runtime flag (5th argv: `0` headless, `1` Pangolin) — no recompile to switch.

## Repository layout

```
01_extract.sh … 06_cull_segment.sh   stage wrappers (+ 03b/05b/06b batched variants)
run_pipeline.sh                       orchestrator (non-batched track)
scripts/NN_*.py                       stage implementations
config/
  camera/RealSense_D456.yaml          camera intrinsics for ORB-SLAM3
  orbslam/                            ORB-SLAM3 configs
  pipeline/default.yaml               pipeline defaults
install/                              dependency / Pangolin / ORB-SLAM3 build + env setup
external/  orbslam3, sam3             git submodules
docs/                                 setup + feature guides
output/<run>/                         gitignored results (meshes, scoring, segments)
```

## Output layout

```
output/<run>/
  sparse/
    CameraTrajectory.txt        ORB-SLAM3 TUM poses
    trajectory_open3d.log       Open3D camera log (used by 03 / 05)
  raw_mesh_rgb.ply              stage 03 geometry mesh
  scoring/
    alpha_maps/alpha_*.npz      L2 EDT cache
    alpha_mesh.ply              stage 05 semantic mesh (grey = EDT score)
  segments/                     stage 06 culled mesh + per-stone submeshes
```

## Documentation
- [Setup guide](docs/SETUP.md) — install, build, environment
- [Segmented reconstruction](docs/SEGMENTED_RECONSTRUCTION.md) · [Boundary segmentation logic](docs/boundary_segmentation_explanation.md)
- [Point-cloud export](docs/POINT_CLOUD_EXPORT.md) · [Viewer toggle](docs/VIEWER_TOGGLE.md) · [Bag trimming](docs/BAG_TRIMMING_GUIDE.md)
- [`CLAUDE.md`](CLAUDE.md) — design decisions, cache layers, tunable parameters

## License
See `LICENSE` if present; otherwise all rights reserved pending a license choice.
