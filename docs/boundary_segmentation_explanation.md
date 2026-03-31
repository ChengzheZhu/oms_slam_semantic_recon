# Boundary-Based Segmentation Logic

## The ID Inconsistency Problem

### Original Approach (03c)
```
Frame 10:  SAM3 detects stone A → ID=1, stone B → ID=2
Frame 20:  SAM3 detects stone B → ID=1, stone C → ID=2, stone A → ID=3
```

**Issue**: The same physical stone gets different IDs in different frames!

During 3D reconstruction:
- Stone A accumulates votes for both ID=1 and ID=3
- Stone B accumulates votes for both ID=1 and ID=2
- Final mesh has conflicting, noisy segment labels

## Boundary-Based Solution (03d)

### Key Insight
> **We don't need to know WHICH stone is which across frames.**
> **We only need to know WHERE the boundaries between stones are.**

### How It Works

#### Stage 1: 2D Boundary Detection (Per Frame)

For each frame:

1. **Run SAM3** to detect objects (local IDs, not tracked)
   ```
   Frame 10: [stone_1, stone_2, stone_3, ...]
   Frame 20: [stone_1, stone_2, stone_4, ...]  ← Different IDs!
   ```

2. **Extract boundaries** between segments using morphological gradient
   ```python
   dilated = grey_dilation(segment_mask)
   eroded = grey_erosion(segment_mask)
   boundary = (dilated != eroded)  # Where segment ID changes
   ```

3. **Result**: Binary boundary mask (True = boundary pixel, False = interior)
   - No IDs propagated!
   - Only boundary locations matter

#### Stage 2: 3D Boundary Accumulation

For each frame:

1. **Project boundary pixels to 3D** using depth + camera pose
   ```
   2D boundary pixel (u,v) + depth(u,v) + camera_pose → 3D point (x,y,z)
   ```

2. **Accumulate all boundary points** across frames into point cloud
   ```
   Frame 10 contributes 5,000 boundary points
   Frame 20 contributes 6,000 boundary points
   ...
   Total: 500,000 boundary points from all frames
   ```

3. **Multi-view fusion**: Same physical boundary seen from multiple angles
   - Reinforces true boundaries (high point density)
   - Reduces noise (spurious detections have low density)

#### Stage 3: Standard TSDF Reconstruction

- Do normal TSDF integration (no segment labels!)
- Just build geometric mesh from RGB-D + poses
- Result: Clean, fused 3D mesh (no segmentation yet)

#### Stage 4: 3D Mesh Segmentation with Boundary Constraints

1. **Classify mesh vertices**
   ```python
   For each vertex in mesh:
       distance = nearest_distance_to_boundary_point
       if distance < boundary_threshold:
           vertex is "on boundary"
       else:
           vertex is "interior"
   ```

2. **Build adjacency graph**
   ```python
   For each triangle edge:
       if both vertices are interior (not on boundary):
           add edge to graph
       else:
           DO NOT add edge (boundary blocks connectivity)
   ```

3. **Connected component analysis** (region growing)
   ```python
   # Find groups of vertices connected by interior edges
   segments = find_connected_components(adjacency_graph)
   # Boundaries act as barriers that separate components
   ```

4. **Extract segment meshes**
   ```python
   For each connected component:
       extract all triangles with vertices in this component
       save as separate mesh file
   ```

## Why This Works

### 1. ID-Free Approach
- No need to track stone identities across frames
- Same stone can have different IDs in different frames → doesn't matter!
- Only geometric boundaries are propagated

### 2. Multi-View Consistency
- Physical boundaries are reinforced by multiple observations
- Noise averages out across views
- 3D geometry naturally groups points from same physical stone

### 3. Spatial Coherence
- Region growing ensures segments are spatially connected
- Boundaries enforce separation between distinct objects
- Natural clustering based on 3D proximity

### 4. Robustness
- Missing detections in some frames → still have boundaries from other frames
- Over-segmentation in 2D → merged in 3D if boundaries don't align spatially
- Under-segmentation in 2D → separated in 3D if other frames provide boundaries

## Visual Analogy

Think of it like painting a wall:

**Old approach (03c)**:
- Number each stone in each photo (1, 2, 3...)
- Try to match numbers across photos (error-prone!)
- Paint stones based on numbers (inconsistent)

**New approach (03d)**:
- Draw lines around stone edges in each photo
- Project all lines onto actual 3D wall
- Lines that appear in many photos are real boundaries
- Fill regions between boundary lines (consistent!)

## Parameters

### `--boundary_threshold` (default: 0.01m = 1cm)
- How close to boundary point cloud must vertex be to count as "on boundary"
- **Lower** (0.005m): Stricter boundaries, may over-segment
- **Higher** (0.02m): Tolerant boundaries, may under-segment

### `--min_cluster_size` (default: 500 triangles)
- Minimum triangles to keep a segment
- **Lower** (100): Keep smaller stones, may have noise
- **Higher** (1000): Only large stones, filters small artifacts

## Example Output

```
Processing 200 frames...
✓ Integration complete

Extracting mesh from TSDF...
  Vertices: 1,234,567
  Triangles: 2,345,678
  Boundary points: 456,789

Segmenting mesh using boundary constraints...
  Boundary vertices: 89,012 / 1,234,567 (7.2%)
  Building adjacency graph...
  Finding connected components...
  Found 47 segments

    Segment 1: 23,456 vertices, 45,678 triangles
    Segment 2: 18,901 vertices, 37,802 triangles
    ...
    Segment 47: 12,345 vertices, 24,690 triangles

✓ Saved 47 segment meshes
```

## Debugging

The script saves `*_boundaries.ply` point cloud for visualization:

```python
# Visualize boundaries
import open3d as o3d

mesh = o3d.io.read_triangle_mesh("output/sam3_boundary_mesh.ply")
boundaries = o3d.io.read_point_cloud("output/sam3_boundary_mesh_boundaries.ply")

# Paint boundaries red for visibility
boundaries.paint_uniform_color([1, 0, 0])

o3d.visualization.draw_geometries([mesh, boundaries])
```

This lets you see where boundaries were detected and verify they align with actual stone edges.


  python scripts/03d_sam3_boundary_reconstruction.py \
    --frames_dir /home/chengzhe/Data/OMS_data3/rs_bags/1101/20251101_235516 \
    --intrinsic /home/chengzhe/Data/OMS_data3/rs_bags/1101/20251101_235516/intrinsic.json \
    --trajectory output/sam3/sparse/trajectory_open3d.log \
    --output output/sam3/sam3_boundary_test.ply \
    --segments_dir output/sam3/sam3_boundary_test_segments \
    --sam_prompt "individual stone" \
    --sam_confidence 0.1 \
    --frame_subsample 5 \
    --voxel_size 0.01


      python scripts/03d_sam3_boundary_reconstruction.py \
    --frames_dir /home/chengzhe/Data/OMS_data3/rs_bags/1101/20251101_235516 \
    --intrinsic /home/chengzhe/Data/OMS_data3/rs_bags/1101/20251101_235516/intrinsic.json \
    --trajectory output/sucess/sparse/trajectory_open3d.log \
    --output output/sam3/sam3_boundary_test.ply \
    --segments_dir output/sam3/sam3_boundary_test_segments \
    --sam_prompt "individual stone" \
    --sam_confidence 0.1 \
    --sam_max_size_ratio 0.15 \
    --frame_subsample 5 \
    --voxel_size 0.01 \
    --boundary_threshold 0.015 \
    --min_cluster_size 200