# Project Structure - Standalone Configuration

## Overview

The `slam_dense_reconstruction` project is now organized as a standalone, modular system with clear separation of concerns.

## Directory Structure

```
slam_dense_reconstruction/
├── bin/                          # Executable scripts
│   └── run_pipeline.py          # Master pipeline runner
│
├── src/                          # Python source modules
│   └── slam_reconstruction/     # Main package
│       └── __init__.py
│
├── config/                       # Configuration files
│   ├── camera/                  # Camera calibrations
│   │   └── RealSense_D456.yaml # D456 camera parameters
│   ├── pipeline/                # Pipeline configurations
│   │   ├── default.yaml        # Default pipeline config
│   │   └── examples/           # Example configs for different datasets
│   │       ├── pole_tall.yaml
│   │       └── trimmed.yaml
│   └── orbslam/                 # ORB_SLAM3 specific configs
│
├── scripts/                      # Utility scripts (legacy/compatibility)
│   ├── 00_extract_frames.py    # Extract frames from bag
│   ├── 01_run_orbslam3.sh      # Run ORB_SLAM3
│   ├── 02_convert_trajectory.py # Convert trajectory format
│   ├── 03_dense_reconstruction.py # Dense TSDF reconstruction
│   └── create_associations.py   # Create TUM associations
│
├── install/                      # Installation helpers
│   ├── install_dependencies.sh  # Install system dependencies
│   └── build_orbslam3.sh        # Build ORB_SLAM3
│
├── external/                     # External dependency info
│   ├── README.md                # Dependency documentation
│   └── orbslam3/                # ORB_SLAM3 integration
│
├── docs/                         # Documentation
│   ├── SETUP.md                 # Installation guide
│   ├── USAGE.md                 # Usage instructions
│   ├── QUICK_START.md           # Quick start guide
│   ├── VIEWER_TOGGLE.md         # Viewer toggle documentation
│   └── BUILD_ORBSLAM3.md        # ORB_SLAM3 build guide
│
├── data/                         # Sample/test data (optional)
│   └── .gitkeep
│
├── output/                       # Generated outputs (gitignored)
│   ├── sparse/                  # SLAM trajectories
│   └── dense/                   # Dense meshes
│
├── tests/                        # Unit tests (future)
│
├── README.md                     # Main project README
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore patterns
└── LICENSE                       # License file (future)
```

## Key Features

### 1. Modular Organization
- **bin/**: Entry points for running the pipeline
- **src/**: Reusable Python modules
- **config/**: All configuration centralized and organized
- **scripts/**: Backward-compatible utilities
- **install/**: Setup and installation helpers

### 2. Clear Dependency Management
- **external/README.md**: Documents all external dependencies
- **requirements.txt**: Python package dependencies
- **install/**: Automated installation scripts

### 3. Flexible Configuration
- **config/camera/**: Camera-specific calibrations
- **config/pipeline/**: Dataset-specific pipeline configs
- **config/pipeline/examples/**: Ready-to-use example configs

### 4. Comprehensive Documentation
- **README.md**: Project overview
- **docs/**: Detailed guides for setup, usage, and features

## Usage Examples

### Using the Reorganized Structure

#### Run with default config:
```bash
cd /home/chengzhe/projects/slam_dense_reconstruction
./bin/run_pipeline.py
```

#### Run with specific dataset config:
```bash
./bin/run_pipeline.py --config config/pipeline/examples/pole_tall.yaml
```

#### Run with command-line override:
```bash
./bin/run_pipeline.py --bag /path/to/new_file.bag --extract
```

### Backward Compatibility

Old scripts still work:
```bash
./scripts/01_run_orbslam3.sh
./scripts/02_convert_trajectory.py
./scripts/03_dense_reconstruction.py
```

## Configuration Hierarchy

1. **Default config**: `config/pipeline/default.yaml`
2. **Camera config**: Referenced from pipeline config
3. **Command-line overrides**: Highest priority

## Dependency Management

### External Dependencies
See `external/README.md` for:
- ORB_SLAM3 location and version
- Required system packages
- Python package versions
- Build instructions

### Installation
```bash
# Install system dependencies
./install/install_dependencies.sh

# Build ORB_SLAM3
./install/build_orbslam3.sh

# Install Python packages
pip install -r requirements.txt
```

## Output Organization

```
output/
├── associations.txt              # Frame associations
├── sparse/                       # ORB_SLAM3 outputs
│   ├── trajectory_tum.txt       # TUM format trajectory
│   ├── trajectory_open3d.log    # Open3D format
│   └── keyframe_trajectory_tum.txt
└── dense/                        # Dense reconstruction
    └── mesh.ply                  # Final mesh
```

## Making it Portable

To share this project:

1. **Archive the project**:
   ```bash
   tar -czf slam_reconstruction.tar.gz \
       --exclude=output \
       --exclude=data \
       --exclude=__pycache__ \
       slam_dense_reconstruction/
   ```

2. **On new system**:
   ```bash
   tar -xzf slam_reconstruction.tar.gz
   cd slam_dense_reconstruction
   ./install/install_dependencies.sh
   ./install/build_orbslam3.sh
   pip install -r requirements.txt
   ```

3. **Configure paths**:
   Edit `config/pipeline/default.yaml` to point to your data

## Version Control

The `.gitignore` excludes:
- `output/` - Generated files
- `data/` - Large datasets
- `*.ply` - Mesh files
- Python cache files
- Build artifacts

## Future Enhancements

- [ ] Add `setup.py` for pip installation
- [ ] Create `tests/` with unit tests
- [ ] Add CI/CD pipeline
- [ ] Docker containerization
- [ ] Add `LICENSE` file
- [ ] Create sample datasets in `data/`
