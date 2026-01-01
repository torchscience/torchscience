# Space Partitioning Phase 2: BVH, Octree, and CUDA (Future Work)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `torchscience.space_partitioning` with additional tree types (BVH, Octree), CUDA backends, and memory-mapped storage.

**Architecture:** Additional tree implementations following the established Phase 1 patterns. CUDA kernels for GPU acceleration. Memory-mapped storage for massive datasets.

**Tech Stack:** PyTorch C++ Extension (libtorch), ATen, CUDA, TensorDict, memory-mapped files.

**Prerequisites:** Phase 1A, 1B, and 1C completed (kd_tree, k_nearest_neighbors, range_search available).

**Status:** PLANNING ONLY - Not ready for implementation.

---

## Planned Features

### Bounding Volume Hierarchy (BVH)

A BVH stores axis-aligned bounding boxes (AABBs) hierarchically. Better than k-d trees for:
- Non-uniform point distributions
- Dynamic scenes (incremental updates)
- Ray tracing applications

**API:**
```python
def bounding_volume_hierarchy(
    points: Tensor,
    *,
    leaf_size: int = 10,
    split_method: str = "sah",  # "sah", "median", "middle"
) -> TensorDict:
    """Build a BVH from points."""
    ...
```

**Tree structure:**
- `aabb_min`: (n_nodes, d) minimum corner of each node's AABB
- `aabb_max`: (n_nodes, d) maximum corner of each node's AABB
- `left`, `right`, `indices`, `leaf_starts`, `leaf_counts` (same as k-d tree)

---

### Octree

An octree recursively subdivides 3D space into 8 octants. Better than k-d trees for:
- Uniform 3D point clouds
- Level-of-detail rendering
- Voxel-based representations

**API:**
```python
def octree(
    points: Tensor,
    *,
    max_depth: int = 10,
    min_points: int = 1,
) -> TensorDict:
    """Build an octree from 3D points."""
    ...
```

**Constraints:**
- Input must be 3D (d=3)
- Points should be normalized to unit cube [0, 1]^3

---

### CUDA Backends

GPU-accelerated versions of all operators:

1. **kd_tree_build (CUDA):** Parallel tree construction
2. **k_nearest_neighbors (CUDA):** Parallel query with warp-level primitives
3. **range_search (CUDA):** Parallel query with dynamic output sizing

**Key challenges:**
- Variable-length output for range_search
- Work stealing for load balancing
- Memory coalescing for tree traversal

---

### Memory-Mapped Storage

Enable trees larger than RAM:

```python
tree = kd_tree(points, mmap=True, path="/tmp/tree.mmap")
```

**Implementation:**
- Use `at::Tensor::from_file()` for memory-mapped tensors
- Lazy loading of tree nodes during traversal
- Explicit `flush()` for persistence

---

## Task Outline (Not Yet Detailed)

### BVH Implementation
1. Add BVH schema definitions
2. Implement BVH CPU kernel with SAH splitting
3. Implement BVH meta and autocast kernels
4. Implement BVH Python wrapper
5. Update k_nearest_neighbors to support BVH
6. Update range_search to support BVH
7. Add BVH tests

### Octree Implementation
1. Add octree schema definitions
2. Implement octree CPU kernel
3. Implement octree meta and autocast kernels
4. Implement octree Python wrapper
5. Update k_nearest_neighbors to support octree
6. Update range_search to support octree
7. Add octree tests

### CUDA Backends
1. Add CUDA kernel infrastructure
2. Implement kd_tree_build CUDA kernel
3. Implement k_nearest_neighbors CUDA kernel
4. Implement range_search CUDA kernel
5. Add CUDA tests

### Memory-Mapped Storage
1. Design memory-mapped tree format
2. Implement mmap support in kd_tree
3. Implement mmap support in queries
4. Add mmap tests

---

## Design Considerations

### BVH vs k-d Tree

| Aspect | k-d Tree | BVH |
|--------|----------|-----|
| Split | Single plane | Box subdivision |
| Balance | Depends on data | SAH optimizes cost |
| Update | Rebuild required | Incremental possible |
| Memory | Smaller (no boxes) | Larger (store AABBs) |
| Ray tracing | Suboptimal | Optimized |

### CUDA Parallelization Strategy

**Tree Construction:**
- Use radix sort for point ordering
- Parallel prefix sum for node allocation
- Bottom-up tree construction

**Query:**
- One warp per query (collaborative traversal)
- Shared memory for node stack
- Atomic operations for output sizing

### Thread Safety

The Phase 1 implementation is thread-safe for:
- Concurrent read queries on the same tree
- Independent trees in different threads

Not thread-safe for:
- Modifying tree during queries
- Concurrent writes to same tree

TensorDict's `lock` parameter can be used for explicit synchronization.

---

## Timeline

**Phase 2 is not yet scheduled.** Priority depends on user demand:
1. CUDA backends (most requested for ML workflows)
2. BVH (useful for ray tracing, graphics)
3. Octree (useful for voxel-based methods)
4. Memory-mapped (useful for massive datasets)

---

## References

- [nanoflann](https://github.com/jlblancoc/nanoflann) - Header-only k-d tree
- [Open3D](http://www.open3d.org/) - Point cloud processing library
- [FAISS](https://github.com/facebookresearch/faiss) - GPU-accelerated similarity search
- [PyTorch3D](https://pytorch3d.org/) - 3D deep learning library
