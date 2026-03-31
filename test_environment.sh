#!/bin/bash

echo "=============================================="
echo "Testing SAM3 + Open3D Environment"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if we're in the right environment
if [ "$CONDA_DEFAULT_ENV" != "sam3_open3d" ]; then
    echo -e "${RED}ERROR: Not in sam3_open3d environment${NC}"
    echo "Please run: conda activate sam3_open3d"
    exit 1
fi

echo -e "${YELLOW}Testing Python version...${NC}"
python -c "import sys; print(f'Python: {sys.version}')"
echo ""

echo -e "${YELLOW}Testing NumPy...${NC}"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')" || echo -e "${RED}FAILED${NC}"
echo ""

echo -e "${YELLOW}Testing PyTorch...${NC}"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" || echo -e "${RED}FAILED${NC}"
echo ""

echo -e "${YELLOW}Testing Open3D...${NC}"
python -c "import open3d as o3d; print(f'Open3D: {o3d.__version__}'); print(f'CUDA support: {o3d.core.cuda.is_available()}'); print(f'GUI support: {hasattr(o3d.visualization, \"gui\")}')" || echo -e "${RED}FAILED${NC}"
echo ""

echo -e "${YELLOW}Testing RealSense...${NC}"
python -c "import pyrealsense2 as rs; ctx = rs.context(); devices = ctx.query_devices(); print(f'pyrealsense2: OK ({len(devices)} device(s) detected)')" || echo -e "${RED}FAILED${NC}"
echo ""

echo -e "${YELLOW}Testing SAM3...${NC}"
python -c "from sam3 import build_sam3_image_model; from sam3.model.sam3_image_processor import Sam3Processor; print('SAM3: OK')" || echo -e "${RED}FAILED - Did you install SAM3 in editable mode?${NC}"
echo ""

echo -e "${YELLOW}Testing SciPy...${NC}"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')" || echo -e "${RED}FAILED${NC}"
echo ""

echo -e "${YELLOW}Testing scikit-learn...${NC}"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')" || echo -e "${RED}FAILED${NC}"
echo ""

echo -e "${YELLOW}Testing matplotlib...${NC}"
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')" || echo -e "${RED}FAILED${NC}"
echo ""

echo -e "${YELLOW}Testing PIL...${NC}"
python -c "from PIL import Image; print(f'Pillow: {Image.__version__}')" || echo -e "${RED}FAILED${NC}"
echo ""

echo "=============================================="
echo -e "${GREEN}Quick Integration Test${NC}"
echo "=============================================="
echo ""

python << 'EOF'
import sys
import torch
import open3d as o3d
import numpy as np
from PIL import Image

print("Creating test point cloud with Open3D...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
print(f"✓ Point cloud created: {len(pcd.points)} points")

print("\nCreating test image with PIL...")
img = Image.new('RGB', (100, 100), color='red')
print(f"✓ Image created: {img.size}")

print("\nCreating test tensor with PyTorch...")
tensor = torch.randn(10, 3)
if torch.cuda.is_available():
    tensor = tensor.cuda()
    print(f"✓ Tensor on GPU: {tensor.device}")
else:
    print(f"✓ Tensor on CPU: {tensor.device}")

print("\n✓ All integration tests passed!")
EOF

echo ""
echo "=============================================="
echo -e "${GREEN}Environment Test Complete!${NC}"
echo "=============================================="
