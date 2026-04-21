1. To increase wall reconstruction density, now setting the voxel size to 0.002 (2mm)
2. For OOM reasons, create a batched processing and storage strategy for the wall mesh
3. Separating the tsdf mesh extraction into batches with overlapping frames. maintain the same structure for sam3 meshes
4. instead of fusing all mesh segemnts together, keep them as separated, and create an association that documents which frames are included in this mesh. This frame is the key to down stream queries.
5. create 1 to 1 rgb meshes and sam3 scroing meshes, then create the culled rgb meshes as segments. Keep those for further use. The meshes should share the same world coordinates (should be solved by the tsdf using orb trajectory already?)

---QR reprojection---
1. after pose relocation, iterate through all culled meshes to find hit clusters
2. perform ICP and mesh fusion for all clusters under the same QR tag to produce high res wall patches
3. send high res wall patches and higher res stone 3d for RAP matching.