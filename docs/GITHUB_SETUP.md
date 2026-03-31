# GitHub Setup Guide

## Important: Git Submodule Note

This project includes ORB_SLAM3 as a git submodule (`external/orbslam3/`). The submodule is already configured and will be included when you push to GitHub.

## Step-by-Step Instructions

### Step 1: Create GitHub Repository (On GitHub Website)

1. Go to https://github.com/new
2. Repository name: `ORB_SLAM3_RGBD_DenseSlamReconstrction` (or your preferred name)
3. Description: `ORB_SLAM3 + Open3D pipeline for RGB-D SLAM and dense 3D reconstruction`
4. Choose **Public** or **Private**
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### Step 2: Verify Local Git Repository

The repository is already initialized with git and includes a submodule. Verify:

```bash
cd /home/chengzhe/projects/ORB_SLAM3_RGBD_DenseSlamReconstrction

# Check git status
git status

# Check submodules
git submodule status
```

You should see `external/orbslam3` as a submodule pointing to your ORB_SLAM3 fork.

### Step 3: Create Initial Commit (if not already done)

```bash
# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: ORB_SLAM3 + Open3D dense reconstruction pipeline

Features:
- ORB_SLAM3 integration (git submodule)
- Open3D TSDF reconstruction
- Configurable YAML pipeline system
- Viewer toggle for batch/interactive modes
- Complete documentation and examples
"
```

### Step 4: Connect to GitHub

Replace `YOUR_USERNAME` with your GitHub username:

```bash
# Add GitHub remote
git remote add origin https://github.com/YOUR_USERNAME/ORB_SLAM3_RGBD_DenseSlamReconstrction.git

# Or using SSH (if you have SSH keys set up):
git remote add origin git@github.com:YOUR_USERNAME/ORB_SLAM3_RGBD_DenseSlamReconstrction.git
```

### Step 5: Push to GitHub

```bash
# Push to main branch (including submodules)
git branch -M main
git push -u origin main
```

The git submodule will be pushed as a reference to your ORB_SLAM3 fork at `https://github.com/ChengzheZhu/ORB_SLAM3.git`.

### Step 6: Verify Upload

1. Go to `https://github.com/YOUR_USERNAME/ORB_SLAM3_RGBD_DenseSlamReconstrction`
2. You should see all your files!
3. Navigate to `external/orbslam3` - it should show as a submodule link to your ORB_SLAM3 fork

## Files That Will Be Included

✅ **bin/** - Pipeline executables
✅ **scripts/** - Pipeline scripts
✅ **config/** - Configuration files (camera & pipeline)
✅ **install/** - Installation scripts
✅ **external/orbslam3** - ORB_SLAM3 submodule (reference only)
✅ **docs/** - All documentation
✅ **README.md** - Project overview
✅ **requirements.txt** - Python dependencies
✅ **.gitignore** - Ignore rules
✅ **.gitmodules** - Submodule configuration

## Files That Will Be EXCLUDED (per .gitignore)

❌ **output/** - Generated trajectories and meshes
❌ **data/** - Large bag files
❌ **external/orbslam3/build/** - ORB_SLAM3 build artifacts
❌ **external/orbslam3/lib/** - ORB_SLAM3 compiled libraries
❌ **\*.ply, \*.pcd** - Mesh/point cloud files
❌ **\*.log** - Log files
❌ **__pycache__/** - Python cache

## Cloning on Another Machine

When others clone your repository, they need to initialize the submodule:

```bash
# Method 1: Clone with submodules
git clone --recursive https://github.com/YOUR_USERNAME/ORB_SLAM3_RGBD_DenseSlamReconstrction.git

# Method 2: Clone then initialize submodules
git clone https://github.com/YOUR_USERNAME/ORB_SLAM3_RGBD_DenseSlamReconstrction.git
cd ORB_SLAM3_RGBD_DenseSlamReconstrction
git submodule update --init --recursive

# Then build
./install/install_dependencies.sh
conda create -n rs_open3d python=3.9
conda activate rs_open3d
pip install -r requirements.txt
./scripts/install_pangolin.sh
./scripts/build_orbslam3.sh
```

## Update Repository After Changes

```bash
# Check status
git status

# Add changed files
git add <files>

# If you updated the submodule:
cd external/orbslam3
git checkout <new-commit>
cd ../..
git add external/orbslam3

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Understanding the Submodule

The `.gitmodules` file contains:

```ini
[submodule "external/orbslam3"]
    path = external/orbslam3
    url = https://github.com/ChengzheZhu/ORB_SLAM3.git
```

This means:
- Your main repository references your ORB_SLAM3 fork
- Only the specific commit hash is stored (not the full code)
- Users must run `git submodule update --init` to download ORB_SLAM3
- ORB_SLAM3 remains a separate repository that you can update independently

## Recommended Repository Settings

### Add Topics (On GitHub)
- `slam`
- `3d-reconstruction`
- `orbslam3`
- `open3d`
- `rgbd`
- `realsense`
- `computer-vision`
- `tsdf`
- `dense-reconstruction`

### Add Badges to README
```markdown
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
```

## Troubleshooting

### Submodule Shows as Modified

If `git status` shows `external/orbslam3` as modified:

```bash
cd external/orbslam3
git status  # Check what changed
git checkout master  # Or the commit you want
cd ../..
git add external/orbslam3
git commit -m "Update ORB_SLAM3 submodule reference"
```

### Large Files Error

If GitHub rejects push due to large files:

```bash
# Find large files
find . -type f -size +50M -not -path "./.git/*"

# Add to .gitignore
echo "path/to/large/file" >> .gitignore
git rm --cached path/to/large/file
git commit -m "Remove large file from tracking"
```

### Authentication Required

**Option 1: Personal Access Token (HTTPS)**
1. GitHub Settings → Developer settings → Personal access tokens
2. Generate token with `repo` scope
3. Use token as password when prompted

**Option 2: SSH Keys**
1. `ssh-keygen -t ed25519 -C "your_email@example.com"`
2. Add public key to GitHub: Settings → SSH and GPG keys
3. Use SSH URL: `git@github.com:YOUR_USERNAME/ORB_SLAM3_RGBD_DenseSlamReconstrction.git`

## Example README Updates

Add installation note about submodules:

```markdown
## Installation

### Clone with Submodules

⚠️ **Important:** This project uses ORB_SLAM3 as a git submodule.

\`\`\`bash
git clone --recursive https://github.com/YOUR_USERNAME/ORB_SLAM3_RGBD_DenseSlamReconstrction.git
\`\`\`

If you forgot `--recursive`:

\`\`\`bash
git submodule update --init --recursive
\`\`\`
```

## Citation

Add to README.md:

```markdown
## Citation

If you use this pipeline in your research, please cite:

\`\`\`bibtex
@software{orbslam3_open3d_pipeline,
  author = {Your Name},
  title = {ORB\_SLAM3 + Open3D Dense Reconstruction Pipeline},
  year = {2024},
  url = {https://github.com/YOUR_USERNAME/ORB_SLAM3_RGBD_DenseSlamReconstrction}
}
\`\`\`

This pipeline uses:
- ORB\_SLAM3: https://github.com/UZ-SLAMLab/ORB_SLAM3
- Open3D: http://www.open3d.org/
```
