# OMS SLAM Semantic Reconstruction — Claude Code Context

## Project overview
Sub-project 2 of 3 in the OMS stone-wall pipeline:
1. **Components** (`oms_monocular_semantic_recon`) — MASt3R/VGGT + SAM3 → per-stone point clouds
2. **This repo** — ORB-SLAM3 RGB-D SLAM + SAM3 seam voting → segmented assembly mesh
3. **Registration** (future) — geometry-based matching of stones into the assembly

RealSense D456 camera. ORB-SLAM3 tracks the camera; Open3D TSDF fuses depth into a dense
mesh; SAM3 masks vote per frame on stone vs seam, fused in a second TSDF.

## Pipeline (7 numbered stages)
Each stage = a root wrapper `NN_*.sh` → `scripts/NN_*.py`. All settings in root `pipeline.yaml`
(key `section.key` → shell var `SECTION_KEY`, loaded by `scripts/pipeline_env.sh`; vars already set
in a wrapper's OVERRIDES block or the environment win). No hard-coded paths in wrappers.
`run_pipeline.sh` runs the non-batched track (`bash run_pipeline.sh [first] [last]`).

| # | Wrapper → script | Does |
|---|------------------|------|
| 01 | `01_extract.sh` → `01_extract_frames.py` | .bag → color/ depth/ confidence/ (depth+conf aligned to colour) intrinsic.json timestamps.txt streams.json |
| 02 | `02_slam.sh` → `02_slam.py` | ORB-SLAM3 RGB-D tracking **and** TUM→Open3D trajectory conversion → trajectory_open3d.log (+ pose-graph JSON) |
| 03 | `03_tsdf_rgb.sh` → `03_tsdf_rgb.py` | TSDF-fuse all frames → raw_mesh_rgb.ply (geometry only) |
| 04 | `04_sam3_mask.sh` → `04_sam3_mask.py` | SAM3 **L1** mask cache; one image-encode, two prompts (stone + paper label); label recovery |
| 05 | `05_sam3_score.sh` → `05_sam3_score.py` | **L2**: per-frame inside mask + EDT → seam votes fused in a semantic TSDF → edt_maps/*.npz + vote_mesh.ply |
| 06 | `06_cull_segment.sh` → `06_cull_segment.py` | KD-tree transfer vote/score→raw mesh, cull seam triangles, segment into stone submeshes |
| 07 | `07_export_rhino.sh` → `07_export_rhino.py` | Rhino export: cloud.ply (mesh vertices = TSDF surface points, face-less PLY → Rhino point cloud) + cameras.obj (OpenCV-convention cone per posed frame `cam_NNNNNN`, trajectory as degree-1 curve). Not glTF: Rhino's glTF importer keeps triangle primitives only. Optional; not in run_pipeline.sh |

**Two tracks:**
- **Non-batched** (`03 → 05 → 06`): single TSDF volume; orchestrated by `run_pipeline.sh`.
- **Batched** (`03b → 05b → 06b`): overlapping temporal batches kept as separate meshes for
  large / high-res (2 mm voxel) scenes; `batch_size`/`batch_overlap` **must match** across 03b & 05b.
  Stages 01/02/04 are shared. Rationale: `docs/drafts/0421_dev_notes.md`.

## Repo layout
```
oms_slam_semantic_recon/
  01_extract.sh 02_slam.sh 03_tsdf_rgb.sh 04_sam3_mask.sh
  05_sam3_score.sh 06_cull_segment.sh 07_export_rhino.sh — stage wrappers
  03b_tsdf_rgb_batched.sh 05b_sam3_score_batched.sh 06b_cull_segment_batched.sh — batched variants
  run_pipeline.sh                                — orchestrator (non-batched)
  pipeline.yaml                                  — global settings: bag, output base, all stage params
  scripts/pipeline_env.sh, config_to_env.py      — YAML → shell vars + derived run paths
  scripts/NN_*.py                                — stage implementations
  install/
    install_dependencies.sh      — system apt packages
    install_pangolin.sh          — builds Pangolin from source
    build_orbslam3.sh            — builds ORB-SLAM3 C++
    setup_env.sh                 — creates slam_recon env + installs SAM3
  external/
    orbslam3/                    — git submodule: ChengzheZhu/ORB_SLAM3.git
    sam3/                        — git submodule: ChengzheZhu/sam3.git (separate from components repo)
  environment.yml                — conda env spec (slam_recon, Python 3.11)
  docs/                          — SETUP.md + feature guides; docs/drafts/ = working notes
```

## Environment
- Conda env: **`slam_recon`** (Python 3.11, PyTorch 2.7+cu126, Open3D 0.19)
- Create with `bash install/setup_env.sh` (env from `environment.yml` + SAM3 editable install)
- SAM3 lives at `external/sam3` — **separate** from the components repo's SAM3
- Version pins: `opencv-python<4.10` (numpy<2 compat), `setuptools<71` (keeps `pkg_resources`),
  `psutil` (SAM3 eagerly imports it)

## ORB-SLAM3 binary
- Built at `external/orbslam3/Examples/RGB-D/rgbd_tum`
- Viewer is a **runtime flag** — 5th argv `0` = headless, `1` = Pangolin viewer (no recompile)
- Vocabulary: `external/orbslam3/Vocabulary/ORBvoc.txt.tar.gz` → extract before first run

## Launching
```bash
conda activate slam_recon

# Full non-batched pipeline (set paths.* in pipeline.yaml first)
bash run_pipeline.sh              # stages 01–06
bash run_pipeline.sh 3 6          # stages 03–06 only

# Single stage
bash 04_sam3_mask.sh

# Batched track (large / high-res) — run stages individually
bash 03b_tsdf_rgb_batched.sh && bash 05b_sam3_score_batched.sh && bash 06b_cull_segment_batched.sh
```

## Scoring model (stages 04–06)
```
04  SAM3 L1 mask cache
    per frame → stone masks + label masks (shared image encoding)
    → <frames_dir>/sam3_<prompt>_conf_<c>/masks_NNNNNN.npz   (one dir per prompt)
    label recovery: label grown label_ring_px; stone covering >= label_min_coverage of
      that perimeter takes the label grown label_grow_px (never into another stone)
    → <frames_dir>/sam3_merged_<stone>+<label>_conf_<c>/masks_NNNNNN.npz (+ params.json)

05  L2 seam votes + semantic TSDF (scripts/vote_common.py)
    per frame: inside = pixel in a stone mask; edt = pixel distance to the mask edge
    seam score s = 2 * clip(d_mm / seam_width_mm, 0, 1) ** seam_gamma - 1   (-1 edge … +1)
      d_mm = (edt - 1) * Z / fx with the pixel's own depth (aligned to colour in step 01)
    TSDF colour R = inside * (s + 1) / 2, G = inside → per voxel after fusion:
      vote  V = avg G             share of the views that put it inside a stone mask
      score S = 2 * R / G - 1     mean s over just those views (shared weights cancel)
    → edt_maps/edt_NNNNNN.npz (L2 cache) + vote_mesh.ply

06  transfer + cull + segment
    cKDTree: vote_mesh vertices → raw_mesh vertices → per-vertex V, S
    seam vertex = V < vote_min (not stone) or S < score threshold (near an edge)
    cull triangles with a seam vertex; segment remainder into stone patches
```

## Cache layers
| Layer | Location | Content | Keyed on |
|-------|----------|---------|----------|
| L1 | `frames_dir/sam3_<prompt>_conf_<c>/masks_NNNNNN.npz` | raw SAM3 masks + scores | prompt (dir name), confidence |
| L1 | `frames_dir/sam3_merged_<stone>+<label>_conf_<c>/masks_NNNNNN.npz` | stone masks with labels merged + scores | prompts (dir name); label settings (`params.json`, rebuilt on change) |
| L2 | `output_dir/scoring/edt_maps/edt_NNNNNN.npz` | inside mask (bool) + pixel EDT (float16) | mask cache + sam_max_size_ratio (`params.json`, rebuilt on change) |

Re-running step 05 with the same masks reuses the L2 cache; seam_width_mm / seam_gamma only
re-run the TSDF. Caches from
before prompt-named dirs (`sam3_mask_cache_conf_<c>`) are not picked up; step 04 prints the
`mv` that reuses one if it was made with the current stone prompt.

## Key design decisions
- **ORB-SLAM3 submodule**: built in-place; vocabulary not in git
- **ORB-SLAM3 settings generated, not stored**: `02_slam.py` renders them from
  `frames/intrinsic.json` (camera K, depth scale) + `pipeline.yaml` `orbslam.*` and writes
  `sparse/orbslam_settings.yaml` beside the atlas. `oms_component_registration` 02a reads it to
  match the atlas camera fx + ORB pyramid + CLAHE (`Preprocess.*` keys, ignored by ORB-SLAM3).
  `Stereo.b` = 0.095 (D456 baseline) and `Camera.RGB` = 0 (imread gives BGR); before 2026-10-21
  these were 0.0387 and 1 (R/B grey weights swapped in the atlas)
- **Shutdown waits for mapping** (fork branch `shutdown-wait`): `System::Shutdown()` drains local
  mapping + loop closing and waits for a running global BA before saving the atlas, up to
  `slam.shutdown_wait_s` (settings key `System.ShutdownWaitSeconds`; absent/0 = old behaviour).
  Upstream saved immediately, so a GBA started near the end of a sequence was lost.
- **Playback is real-time**: `rgbd_tum` sleeps to each frame's real timestamp gap (headless or
  background alike); `slam.fps` does NOT change speed — it is ORB-SLAM3's max frames between KFs.
- **Fork branch `shutdown-wait`** (off `66911af`) also fixes `SaveTrajectoryTUM` segfaulting after
  tracking loss (null parent KF) and `LoadAtlas` failing on absolute paths. Fork `master` diverges:
  it lowers relocalisation thresholds for all modes (50→20 PnP inliers) — not merged here.
- **Poses matched to frames by timestamp**: `trajectory_open3d.log` is `timestamp + 4×4 c2w` per
  tracked frame; `pipeline_common.load_frame_poses` pairs each frame's `timestamps.txt` entry with
  the nearest pose (≤ 1 ms) and steps 03/03b/05/05b skip frames with none. ORB-SLAM3 omits lost
  frames and, while tracking is recently lost, repeats the last pose under the last timestamp;
  `02_slam.py` drops those repeats. Index pairing (pre-2026-10-21) shifted every pose after the
  first lost frame. Old 16-column logs: used only if pose count == frame count, else regenerate
  with `python scripts/02_slam.py --convert_only --frames_dir … --output_dir <sparse dir>`.
- **Runtime viewer toggle**: `rgbd_tum.cc` accepts argv[5] (0=headless, 1=viewer)
- **FRAME_STRIDE matters**: stride=1 (30 fps) is more reliable than stride=3 for tracking on fast motion
- **Seam voting** (2026-10-21; replaced the EDT alpha `(dist/max_dist)**edt_gamma`, culled at
  0.35): the old alpha averaged "outside every mask" as 0, so a stone SAM3 missed in some views,
  or a label merged in only some views, was pulled toward the cut the same way a seam is. The vote
  V (is it stone) and score S (how far from an edge, over just the views that say stone) separate
  the two. Seam width is in mm, not a share of each mask's largest distance, so it no longer
  scales with how large or how cut-off a stone looks in a frame
- **CPU TSDF**: `ScalableTSDFVolume` (Open3D legacy API); GPU `VoxelBlockGrid` disabled (v0.19 device issues)
- **Batched track**: keeps TSDF batches as separate PLYs to hit 2 mm voxel within ~30 GB RAM;
  scoring + culling operate on one batch mesh at a time (see `docs/drafts/0421_dev_notes.md`)
- **Label recovery** (step 04): paper labels on stones read as holes or notches in the stone
  mask; SAM3 `sam3.label_prompt` finds them and the stone covering most of the label's grown
  perimeter takes it. Replaced the "QR code marker" prompt + 20 px ring (2026-10-21): QR masks
  missed the paper around the code, and the wide ring reached past small stones
- **Masks cut by the frame edge are kept**: the EDT does not treat the image edge as background,
  so they add no seam there, and the sweep fuses each region from many views. Dropping them
  (tested 2026-10-21, 300-frame sweep) lost 3.9% of the surface and culled 2.1% more along seams
- **Run layout**: everything for one bag under `<paths.output_base>/<run_name>/` (frames/ included);
  set `paths.frames_dir` to reuse frames stored elsewhere

## On a new machine
```bash
git clone --recurse-submodules https://github.com/ChengzheZhu/oms_slam_semantic_recon.git
cd oms_slam_semantic_recon
sudo bash install/install_dependencies.sh
bash install/install_pangolin.sh
bash install/build_orbslam3.sh
cd external/orbslam3/Vocabulary && tar -xf ORBvoc.txt.tar.gz && cd -
bash install/setup_env.sh          # creates slam_recon env + installs SAM3
conda activate slam_recon
bash run_pipeline.sh               # set paths.* in pipeline.yaml first
```

## Bag files
- RealSense D456, RGB-D (no confidence stream in current recordings)
- Bags are large (2–8 GB); run locally for stride=1 tests

## Data layout
Output written to `<paths.output_base>/<run_name>/` (default `output/`, gitignored):
```
<output_base>/<run_name>/
  frames/                         — stage 01 frames + intrinsic.json (+ stage 04 SAM3 caches)
  sparse/
    CameraTrajectory.txt          — ORB-SLAM3 TUM poses
    trajectory_open3d.log         — timestamp + 4×4 c2w per tracked frame (03 / 05 match by timestamp)
  raw_mesh_rgb.ply                — stage 03 geometry mesh
  scoring/
    edt_maps/edt_*.npz            — L2 EDT cache
    vote_mesh.ply                 — stage 05 semantic mesh (colour R, G = score, vote)
    vote_batches/                 — batched track (05b)
  batches/                        — batched track (03b)
  segments/                       — stage 06 culled mesh + per-stone submeshes
  export/                         — stage 07 cloud.ply + cameras.obj for Rhino (SLAM world frame)
```

## Tunable parameters
All live in `pipeline.yaml` (full list + comments there). Key ones:
| Where | YAML key (shell var) | Effect |
|-------|----------------------|--------|
| 04 | `sam3.label_ring_px`, `sam3.label_min_coverage`, `sam3.label_grow_px` | Label → stone assignment + merged area |
| 05 | `score.seam_width_mm` (`SCORE_SEAM_WIDTH_MM`) | Seam score reaches +1 this far inside a mask (mm) |
| 05 | `score.seam_gamma` | Score falloff shape; <1 narrower seam |
| 05 | `score.sam_max_size_ratio` | Max mask size fraction kept |
| 03/05 | `tsdf.voxel_size` (`TSDF_VOXEL_SIZE`) | TSDF resolution (m); smaller = finer/slower |
| 06 | `cull.score_thresholds` | Seam score cutoff sweep (S in [-1, 1]) |
| 06 | `cull.vote_min` | Min share of views calling a vertex stone |
| 06 | `cull.mesh_keep_components` | Keep N largest components |
| 06 | `cull.min_cluster_size` | Min triangles per saved segment |
| 03b/05b | `batch.size`, `batch.overlap` | Batched-track chunking (one setting, shared by 03b & 05b) |
