#!/usr/bin/env python3
"""
Merge pre-processed segment meshes into a single mesh.

Useful for merging hand-edited segment meshes after manual cleanup
(e.g., removing foliage, artifacts, etc.)

Usage:
    # Merge all segments in a directory
    ./merge_segment_meshes.py --input output/run/dense/segments --output merged.ply

    # Merge specific segment files
    ./merge_segment_meshes.py --files seg1.ply seg2.ply seg3.ply --output merged.ply

    # Skip deduplication (faster, but may have duplicate vertices)
    ./merge_segment_meshes.py --input segments/ --output merged.ply --no-deduplicate
"""

import open3d as o3d
import argparse
from pathlib import Path
import sys


def load_segment_meshes(segment_files):
    """Load all segment meshes from file list."""
    meshes = []
    has_colors_count = 0

    print(f"\nLoading {len(segment_files)} segment meshes...")

    for i, seg_file in enumerate(segment_files, 1):
        if not Path(seg_file).exists():
            print(f"  ⚠ Warning: File not found: {seg_file}")
            continue

        mesh = o3d.io.read_triangle_mesh(str(seg_file))

        if len(mesh.vertices) == 0:
            print(f"  ⚠ Warning: Empty mesh: {seg_file}")
            continue

        file_size_mb = Path(seg_file).stat().st_size / (1024 * 1024)
        has_colors = mesh.has_vertex_colors()
        if has_colors:
            has_colors_count += 1

        color_str = "with colors" if has_colors else "no colors"
        print(f"  [{i}/{len(segment_files)}] {Path(seg_file).name}: "
              f"{len(mesh.vertices):,} vertices, "
              f"{len(mesh.triangles):,} triangles, "
              f"{color_str} "
              f"({file_size_mb:.1f} MB)")

        meshes.append(mesh)

    if len(meshes) == 0:
        raise ValueError("No valid meshes found to merge!")

    print(f"\n✓ Loaded {len(meshes)} valid segment meshes")
    if has_colors_count > 0:
        print(f"  {has_colors_count}/{len(meshes)} segments have vertex colors")
    else:
        print(f"  ⚠ Warning: None of the segments have vertex colors!")

    return meshes


def merge_meshes(meshes, deduplicate=True, tolerance=1e-6, remove_duplicated_triangles=True,
                 remove_degenerate=True, remove_non_manifold=False):
    """
    Merge multiple meshes into one.

    Args:
        meshes: List of Open3D TriangleMesh objects
        deduplicate: Remove duplicate vertices after merging
        tolerance: Distance threshold for duplicate vertices (in meters)
        remove_duplicated_triangles: Remove duplicate triangles
        remove_degenerate: Remove degenerate triangles
        remove_non_manifold: Remove non-manifold edges (more aggressive cleaning)

    Returns:
        merged: Single merged TriangleMesh
    """
    print(f"\nMerging {len(meshes)} meshes...")

    if len(meshes) == 1:
        print("  Only one mesh, returning as-is")
        return meshes[0]

    # Merge by concatenating
    merged = meshes[0]
    for i, mesh in enumerate(meshes[1:], start=2):
        merged += mesh
        print(f"  Merged {i}/{len(meshes)} meshes", end='\r')

    print(f"\n\nMerged mesh statistics (before cleaning):")
    print(f"  Total vertices: {len(merged.vertices):,}")
    print(f"  Total triangles: {len(merged.triangles):,}")

    # Clean up overlapping geometry
    if deduplicate:
        print(f"\n  Removing duplicate vertices (tolerance: {tolerance}m)...")
        # Note: Open3D's remove_duplicated_vertices doesn't accept tolerance in older versions
        # We need to use a workaround
        import numpy as np
        from scipy.spatial import cKDTree

        vertices = np.asarray(merged.vertices)
        triangles = np.asarray(merged.triangles)

        # Check if mesh has vertex colors
        has_colors = merged.has_vertex_colors()
        if has_colors:
            vertex_colors = np.asarray(merged.vertex_colors)

        # Build KD-tree for fast nearest neighbor search
        tree = cKDTree(vertices)

        # Find duplicates within tolerance
        pairs = tree.query_pairs(tolerance)

        if len(pairs) > 0:
            print(f"    Found {len(pairs)} duplicate pairs within {tolerance}m")

            # Create mapping from old to new vertex indices
            vertex_map = np.arange(len(vertices))
            for i, j in pairs:
                # Map both to the smaller index
                min_idx = min(vertex_map[i], vertex_map[j])
                vertex_map[i] = min_idx
                vertex_map[j] = min_idx

            # Remap triangles to use deduplicated vertices
            triangles = vertex_map[triangles]

            # Keep only unique vertices
            unique_indices = np.unique(vertex_map)
            inverse_map = np.zeros(len(vertices), dtype=int)
            inverse_map[unique_indices] = np.arange(len(unique_indices))

            # Deduplicate vertices and colors
            vertices = vertices[unique_indices]
            if has_colors:
                vertex_colors = vertex_colors[unique_indices]

            # Remap triangles
            triangles = inverse_map[triangles]

            # Rebuild mesh with colors preserved
            merged = o3d.geometry.TriangleMesh()
            merged.vertices = o3d.utility.Vector3dVector(vertices)
            merged.triangles = o3d.utility.Vector3iVector(triangles)

            if has_colors:
                merged.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                print(f"    Preserved vertex colors")

            print(f"    After deduplication: {len(merged.vertices):,} vertices")
        else:
            print(f"    No duplicates found within tolerance")

    # Remove duplicate triangles
    if remove_duplicated_triangles:
        print(f"\n  Removing duplicate triangles...")
        original_tri_count = len(merged.triangles)
        merged = merged.remove_duplicated_triangles()
        removed = original_tri_count - len(merged.triangles)
        if removed > 0:
            print(f"    Removed {removed:,} duplicate triangles")
        else:
            print(f"    No duplicate triangles found")

    # Remove degenerate triangles
    if remove_degenerate:
        print(f"\n  Removing degenerate triangles...")
        original_tri_count = len(merged.triangles)
        merged = merged.remove_degenerate_triangles()
        removed = original_tri_count - len(merged.triangles)
        if removed > 0:
            print(f"    Removed {removed:,} degenerate triangles")
        else:
            print(f"    No degenerate triangles found")

    # Remove non-manifold edges (more aggressive)
    if remove_non_manifold:
        print(f"\n  Removing non-manifold edges...")
        original_tri_count = len(merged.triangles)
        merged = merged.remove_non_manifold_edges()
        removed = original_tri_count - len(merged.triangles)
        if removed > 0:
            print(f"    Removed {removed:,} triangles with non-manifold edges")
        else:
            print(f"    No non-manifold edges found")

    print(f"\nFinal mesh statistics:")
    print(f"  Vertices: {len(merged.vertices):,}")
    print(f"  Triangles: {len(merged.triangles):,}")
    print(f"  Has vertex colors: {merged.has_vertex_colors()}")

    # Recompute normals
    print("\n  Recomputing vertex normals...")
    merged.compute_vertex_normals()

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge segment meshes into a single mesh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all .ply files in a directory
  %(prog)s --input output/run/dense/segments --output merged.ply

  # Merge specific files (useful after manual cleanup)
  %(prog)s --files segment_01_cleaned.ply segment_02_cleaned.ply --output merged.ply

  # Merge with pattern matching
  %(prog)s --input segments/ --pattern "segment_*.ply" --output merged.ply

  # Skip deduplication for faster merging
  %(prog)s --input segments/ --output merged.ply --no-deduplicate
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str,
                            help='Directory containing segment mesh files')
    input_group.add_argument('--files', nargs='+', type=str,
                            help='List of specific mesh files to merge')

    # Output and options
    parser.add_argument('--output', type=str, required=True,
                       help='Output merged mesh file (.ply)')
    parser.add_argument('--pattern', type=str, default='segment_*.ply',
                       help='File pattern to match in input directory (default: segment_*.ply)')
    parser.add_argument('--sort', action='store_true',
                       help='Sort segment files alphabetically before merging')

    # Cleaning options
    parser.add_argument('--no-deduplicate', action='store_true',
                       help='Skip removing duplicate vertices (faster)')
    parser.add_argument('--tolerance', type=float, default=0.001,
                       help='Distance threshold for duplicate vertices in meters (default: 0.001 = 1mm)')
    parser.add_argument('--no-remove-dup-triangles', action='store_true',
                       help='Skip removing duplicate triangles')
    parser.add_argument('--no-remove-degenerate', action='store_true',
                       help='Skip removing degenerate triangles')
    parser.add_argument('--remove-non-manifold', action='store_true',
                       help='Remove non-manifold edges (aggressive, may remove valid geometry)')

    args = parser.parse_args()

    print("="*80)
    print("Segment Mesh Merger")
    print("="*80)

    # Get list of segment files
    if args.input:
        # Load from directory with pattern
        input_dir = Path(args.input)
        if not input_dir.exists():
            print(f"\n❌ Error: Directory not found: {args.input}")
            return 1

        segment_files = sorted(input_dir.glob(args.pattern))

        if len(segment_files) == 0:
            print(f"\n❌ Error: No files matching pattern '{args.pattern}' in {args.input}")
            return 1

        print(f"\nInput directory: {args.input}")
        print(f"Pattern: {args.pattern}")
        print(f"Found {len(segment_files)} segment files")
    else:
        # Load from file list
        segment_files = [Path(f) for f in args.files]
        print(f"\nInput files: {len(segment_files)} specified")

    # Sort if requested
    if args.sort:
        segment_files = sorted(segment_files)

    # Load meshes
    try:
        meshes = load_segment_meshes(segment_files)
    except Exception as e:
        print(f"\n❌ Error loading meshes: {e}")
        return 1

    # Merge meshes
    try:
        merged = merge_meshes(
            meshes,
            deduplicate=not args.no_deduplicate,
            tolerance=args.tolerance,
            remove_duplicated_triangles=not args.no_remove_dup_triangles,
            remove_degenerate=not args.no_remove_degenerate,
            remove_non_manifold=args.remove_non_manifold
        )
    except Exception as e:
        print(f"\n❌ Error merging meshes: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save merged mesh
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving merged mesh: {args.output}")
    o3d.io.write_triangle_mesh(str(output_path), merged)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Mesh saved ({file_size_mb:.1f} MB)")

    print("\n" + "="*80)
    print("Merge Complete!")
    print("="*80)
    print(f"\nOutput: {args.output}")
    print(f"  Vertices: {len(merged.vertices):,}")
    print(f"  Triangles: {len(merged.triangles):,}")
    print(f"  Has colors: {merged.has_vertex_colors()}")
    print(f"  Size: {file_size_mb:.1f} MB")

    print("\nVisualize with:")
    print(f'  python -c "import open3d as o3d; mesh = o3d.io.read_triangle_mesh(\'{args.output}\'); o3d.visualization.draw_geometries([mesh])"')

    if not merged.has_vertex_colors():
        print("\n⚠ Note: Output mesh has no vertex colors.")
        print("   This may be because input segments didn't have colors,")
        print("   or colors were lost during manual editing.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
