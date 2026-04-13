#!/usr/bin/env bash
# Create the slam_recon conda environment and install all dependencies.
# Run once from the project root:
#   bash install/setup_env.sh
#
# After this completes, activate with:
#   conda activate slam_recon

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_NAME="slam_recon"
SAM3_REPO="https://github.com/ChengzheZhu/sam3.git"
SAM3_DIR="$PROJECT_DIR/external/sam3"

# ── conda ─────────────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh

echo "════════════════════════════════════════"
echo " Creating conda env: $ENV_NAME"
echo "════════════════════════════════════════"

conda env create -f "$PROJECT_DIR/environment.yml" --name "$ENV_NAME"

# ── SAM3 checkout ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " Cloning SAM3 → external/sam3"
echo "════════════════════════════════════════"

if [ ! -d "$SAM3_DIR/.git" ]; then
    git clone "$SAM3_REPO" "$SAM3_DIR"
else
    echo "  SAM3 already cloned, skipping."
fi

# ── SAM3 editable install ─────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " Installing SAM3 (editable)"
echo "════════════════════════════════════════"

conda run -n "$ENV_NAME" pip install -e "$SAM3_DIR"

# ── verify ────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " Verifying key packages"
echo "════════════════════════════════════════"

conda run -n "$ENV_NAME" python -c "
import torch, open3d, cv2, pyrealsense2, sam3, numpy
print(f'  torch        {torch.__version__}  (CUDA: {torch.cuda.is_available()})')
print(f'  open3d       {open3d.__version__}')
print(f'  opencv       {cv2.__version__}')
print(f'  numpy        {numpy.__version__}')
print(f'  pyrealsense2 ok')
print(f'  sam3         {sam3.__version__}')
"

echo ""
echo "✓ Environment ready. Activate with:"
echo "  conda activate $ENV_NAME"
