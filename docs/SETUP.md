# Setup Guide

Full install for the SLAM semantic reconstruction pipeline: system libraries,
ORB-SLAM3 build, and the `slam_recon` conda environment (Python 3.11, PyTorch
2.7 + cu126, Open3D 0.19, SAM3).

Tested on Ubuntu 20.04+. GPU recommended (SAM3 inference + Open3D).

## 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/ChengzheZhu/oms_slam_semantic_recon.git
cd oms_slam_semantic_recon
# If you already cloned without --recurse-submodules:
git submodule update --init --recursive
```

Submodules: `external/orbslam3` (ORB-SLAM3 fork) and `external/sam3` (SAM3 fork,
separate from the components repo's SAM3).

## 2. System dependencies

```bash
sudo bash install/install_dependencies.sh
```

Installs build tools + Eigen, OpenCV, RealSense SDK (`librealsense2`), GTK/GL libs.

## 3. Pangolin (ORB-SLAM3 viewer)

```bash
bash install/install_pangolin.sh
```

## 4. Build ORB-SLAM3

```bash
bash install/build_orbslam3.sh
```

Builds the third-party libs (DBoW2, g2o, Sophus), the core library, and the RGB-D
example binary at `external/orbslam3/Examples/RGB-D/rgbd_tum`. The viewer is a
**runtime flag** (5th argv: `0` headless, `1` Pangolin) — no recompile to switch.

## 5. ORB vocabulary

```bash
cd external/orbslam3/Vocabulary && tar -xf ORBvoc.txt.tar.gz && cd -
```

The vocabulary is large and not tracked in git; extract it before the first SLAM run.

## 6. Python environment

```bash
bash install/setup_env.sh          # creates the slam_recon env + installs SAM3 editable
conda activate slam_recon
```

`install/setup_env.sh` builds the env from `environment.yml` and `pip install -e`s
SAM3 from `external/sam3`. Key version pins (already in `environment.yml`):
- `opencv-python<4.10` — for `numpy<2` (SAM3 requires numpy<2)
- `setuptools<71` — ≥72 drops the top-level `pkg_resources` module
- `psutil` — SAM3 transitive dep (eagerly imported)

## 7. Verify

```bash
conda activate slam_recon
# ORB-SLAM3 binary + vocabulary present
ls -lh external/orbslam3/Examples/RGB-D/rgbd_tum
ls -lh external/orbslam3/Vocabulary/ORBvoc.txt
# Python stack
python -c "import open3d, numpy, cv2, torch, pyrealsense2; \
print('open3d', open3d.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

Then run the pipeline (see the [README](../README.md)):
```bash
bash run_pipeline.sh
```

## Troubleshooting

- **NumPy 2 conflict** (`numpy.core.multiarray failed to import`): SAM3/Open3D need
  `numpy<2` — `pip install "numpy<2" --force-reinstall` in the `slam_recon` env.
- **CUDA not available**: check `nvidia-smi` / `nvcc --version`; reinstall torch with
  `--index-url https://download.pytorch.org/whl/cu126`.
- **ORB-SLAM3 build fails**: rebuild verbose —
  `cd external/orbslam3 && rm -rf build && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON && make -j$(nproc) VERBOSE=1`.
- **Pangolin libs not found**: `export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH`
  (add to `~/.bashrc` to persist).
- **Open3D GUI over SSH**: use headless rendering (`OPEN3D_HEADLESS=1`) or run stages
  with the `--headless` flag where available.
