# SAM3 + Open3D Environment Setup Guide

This guide helps you create a conda environment that supports both SAM3 segmentation and Open3D TSDF reconstruction with RealSense support.

## Quick Start

```bash
cd /home/chengzhe/projects/_OMS/ORB_SLAM3_RGBD_DenseSlamReconstrction

# Create the environment
conda env create -f environment_sam3_open3d.yml

# Activate it
conda activate sam3_open3d

# Install SAM3 in editable mode
cd /home/chengzhe/projects/sam3
pip install -e .

# Test the environment
cd /home/chengzhe/projects/_OMS/ORB_SLAM3_RGBD_DenseSlamReconstrction
bash test_environment.sh
```

## What's Included

### Core Components
- **Python 3.13**: Latest stable Python
- **PyTorch 2.7.0 + CUDA 12.6**: Deep learning framework with GPU support
- **Open3D ≥0.18.0**: 3D data processing and TSDF reconstruction
- **pyrealsense2**: Intel RealSense camera SDK
- **SAM3**: Segment Anything Model 3 for semantic segmentation

### Scientific Computing
- NumPy < 2.0 (for Open3D compatibility)
- SciPy
- scikit-learn
- scikit-image

### Visualization
- Matplotlib
- Pillow
- Jupyter
- IPython

### Additional Tools
- TEASER++ dependencies (pybind11)
- OpenCV
- PyCocoTools
- Einops

## Detailed Setup Instructions

### Step 1: Create Environment

```bash
conda env create -f environment_sam3_open3d.yml
```

This will:
- Download and install all conda packages
- Set up CUDA toolkit
- Install Qt6 for Open3D GUI
- Install scientific computing libraries

**Time**: ~5-10 minutes depending on internet speed

### Step 2: Activate Environment

```bash
conda activate sam3_open3d
```

You should see `(sam3_open3d)` in your terminal prompt.

### Step 3: Install SAM3

SAM3 needs to be installed in editable mode to allow local modifications:

```bash
cd /home/chengzhe/projects/sam3
pip install -e .
```

This will:
- Install SAM3 from your local copy
- Allow you to modify SAM3 code without reinstalling
- Download SAM3 model weights from Hugging Face

**Note**: First run may take time to download model weights (~1-2GB)

### Step 4: Test Installation

Run the comprehensive test script:

```bash
cd /home/chengzhe/projects/_OMS/ORB_SLAM3_RGBD_DenseSlamReconstrction
bash test_environment.sh
```

This will test:
- Python version
- NumPy, PyTorch, Open3D, RealSense, SAM3
- CUDA availability
- GUI support
- Integration between libraries

Expected output:
```
Python: 3.13.x
NumPy: 1.x.x
PyTorch: 2.7.0
CUDA available: True
Open3D: 0.18.x
CUDA support: True
pyrealsense2: 2.x.x
SAM3: OK
✓ All integration tests passed!
```

## Verifying Specific Features

### Test TSDF Reconstruction

```python
import open3d as o3d
import numpy as np

# Create TSDF volume
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.005,
    sdf_trunc=0.02,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)
print("✓ TSDF volume created successfully")
```

### Test SAM3 Segmentation

```python
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Check CUDA
if torch.cuda.is_available():
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    print("✓ CUDA enabled with bfloat16")

# Load model
model = build_sam3_image_model()
processor = Sam3Processor(model, confidence_threshold=0.1)
print("✓ SAM3 model loaded successfully")
```

### Test RealSense

```python
import pyrealsense2 as rs

# List connected devices
ctx = rs.context()
devices = ctx.query_devices()
print(f"✓ RealSense devices found: {len(devices)}")
```

## Common Issues and Solutions

### Issue 1: NumPy Version Conflict

**Error**: `ImportError: numpy.core.multiarray failed to import`

**Solution**: Ensure NumPy < 2.0 is installed
```bash
conda activate sam3_open3d
pip install "numpy<2.0" --force-reinstall
```

### Issue 2: CUDA Not Available

**Error**: `torch.cuda.is_available() returns False`

**Solution**:
1. Check NVIDIA driver: `nvidia-smi`
2. Verify CUDA toolkit: `nvcc --version`
3. Reinstall PyTorch with correct CUDA version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Issue 3: SAM3 Import Error

**Error**: `ModuleNotFoundError: No module named 'sam3'`

**Solution**: Install SAM3 in editable mode
```bash
cd /home/chengzhe/projects/sam3
pip install -e .
```

### Issue 4: Open3D GUI Not Working

**Error**: Open3D visualization crashes or doesn't display

**Solutions**:
1. Check if running over SSH without X11 forwarding - use offscreen rendering
2. Verify Qt6 installation: `conda list | grep qt6`
3. Try headless mode:
```python
import os
os.environ['OPEN3D_HEADLESS'] = '1'
```

### Issue 5: RealSense Library Not Found

**Error**: `ImportError: librealsense2.so`

**Solution**: Reinstall pyrealsense2
```bash
pip uninstall pyrealsense2
pip install pyrealsense2 --no-cache-dir
```

## Environment Management

### List All Environments
```bash
conda env list
```

### Remove Environment
```bash
conda deactivate
conda env remove -n sam3_open3d
```

### Export Environment (for sharing)
```bash
conda activate sam3_open3d
conda env export --no-builds > my_environment.yml
```

### Update Environment
```bash
conda activate sam3_open3d
conda update --all
```

## Using the Environment

### For Reconstruction Scripts

```bash
conda activate sam3_open3d
cd /home/chengzhe/projects/_OMS/ORB_SLAM3_RGBD_DenseSlamReconstrction

# Run boundary-based reconstruction
python scripts/03d_sam3_boundary_reconstruction.py \
  --frames_dir /path/to/frames \
  --intrinsic /path/to/intrinsic.json \
  --trajectory /path/to/trajectory.log \
  --output output/mesh.ply \
  --segments_dir output/segments
```

### For Registration Scripts

```bash
conda activate sam3_open3d
cd /home/chengzhe/projects/_OMS/Component_Registration

# Run stone matching
python 11_test_single_stone.py
```

### In Jupyter Notebook

```bash
conda activate sam3_open3d
jupyter notebook
```

Then import as normal:
```python
import open3d as o3d
from sam3 import build_sam3_image_model
import torch
```

## Performance Optimization

### Enable CUDA for PyTorch
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### Enable bfloat16 for SAM3
```python
if torch.cuda.is_available():
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

### Open3D CUDA
Open3D 0.18+ has CUDA support for certain operations. Check with:
```python
import open3d as o3d
print(f"Open3D CUDA available: {o3d.core.cuda.is_available()}")
```

## Next Steps

1. ✓ Create environment
2. ✓ Test installation
3. Run SAM3 reconstruction: `python scripts/03d_sam3_boundary_reconstruction.py`
4. Run stone matching: `python 11_test_single_stone.py` or `python 11_match_rough_to_fine.py`

## Support

If you encounter issues not covered here:
1. Check conda environment is activated: `echo $CONDA_DEFAULT_ENV`
2. Verify package versions: `conda list | grep -E "(torch|open3d|numpy)"`
3. Run test script: `bash test_environment.sh`
4. Check GPU: `nvidia-smi`

## Differences from Existing Environments

### vs `sam3` environment
- ✓ Adds Open3D
- ✓ Adds pyrealsense2
- ✓ Ensures NumPy < 2.0 for compatibility
- ✓ Adds SciPy, scikit-learn for registration

### vs `rs_open3d` environment
- ✓ Adds SAM3 and all dependencies
- ✓ Adds PyTorch with CUDA
- ✓ More complete visualization stack

This environment combines the best of both worlds!
