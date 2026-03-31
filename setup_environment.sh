#!/bin/bash
set -e

echo "=============================================="
echo "SAM3 + Open3D Environment Setup"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}ERROR: conda not found. Please install Anaconda or Miniconda first.${NC}"
    exit 1
fi

echo -e "${GREEN}Step 1: Creating conda environment from YAML...${NC}"
conda env create -f environment_sam3_open3d.yml

echo ""
echo -e "${GREEN}Step 2: Activating environment...${NC}"
# Note: This won't work in a script, user needs to do it manually
echo -e "${YELLOW}Please run: conda activate sam3_open3d${NC}"

echo ""
echo -e "${GREEN}Step 3: Installing SAM3 in editable mode...${NC}"
echo -e "${YELLOW}After activating the environment, run:${NC}"
echo "  cd /home/chengzhe/projects/sam3"
echo "  pip install -e ."

echo ""
echo -e "${GREEN}Step 4: Testing Open3D installation...${NC}"
echo -e "${YELLOW}Test with:${NC}"
echo "  python -c 'import open3d as o3d; print(f\"Open3D: {o3d.__version__}\")'"

echo ""
echo -e "${GREEN}Step 5: Testing RealSense installation...${NC}"
echo -e "${YELLOW}Test with:${NC}"
echo "  python -c 'import pyrealsense2 as rs; print(f\"RealSense: {rs.__version__}\")'"

echo ""
echo -e "${GREEN}Step 6: Testing SAM3 installation...${NC}"
echo -e "${YELLOW}Test with:${NC}"
echo "  python -c 'from sam3 import build_sam3_image_model; print(\"SAM3 OK\")'"

echo ""
echo -e "${GREEN}Step 7: Testing PyTorch CUDA...${NC}"
echo -e "${YELLOW}Test with:${NC}"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.cuda.is_available()}\")'"

echo ""
echo "=============================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  conda activate sam3_open3d"
echo ""
echo "To test everything at once, run:"
echo "  bash test_environment.sh"
