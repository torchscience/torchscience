# Design: `torchscience.space_partitioning`

**Date:** 2025-12-31
**Status:** Approved

## Overview

Spatial data structures for efficient neighbor search, range queries, and spatial indexing. The module provides the data structures and their query operations — higher-level geometric algorithms live in `torchscience.geometry`.

### Scope

**In scope:**
- k-d tree — Low-dimensional k-NN and range queries
- Bounding volume hierarchy — For ray tracing and collision
- Octree — 3D uniform spatial partitioning

**Out of scope (for now):**
- Ball tree
- Spatial hashing
- R-tree
- BSP tree

### Design Principles

| Principle | Decision |
|-----------|----------|
| API style | Functional (not OO) |
| State representation | TensorDict |
| Variable-length outputs | Nested tensors via TensorDict |
| Distance metrics | Minkowski family (p parameter) |
| Gradients | Through distances only (indices are discrete) |
| Device support | CPU-only for MVP |

### Dependency

Adds `tensordict` as a required dependency.

---

## API Design

### Build Functions (separate)

Each structure has its own build function:

```python
def kd_tree(
    points: Tensor,  # (n, d)
    *,
    leaf_size: int = 10,
) -> TensorDict:
    """Build a k-d tree from points.

    Parameters
    ----------
    points : Tensor, shape (n, d)
        Points to index. Any dimensionality d.
    leaf_size : int, default=10
        Maximum points per leaf node.

    Returns
    -------
    TensorDict
        Tree structure with '_type' = 'kd_tree'.
    """
```

```python
def bounding_volume_hierarchy(
    bounds: Tensor,  # (n, 2, d)
    *,
    leaf_size: int = 4,
    build_method: str = "sah",
) -> TensorDict:
    """Build a bounding volume hierarchy.

    Parameters
    ----------
    bounds : Tensor, shape (n, 2, d)
        Axis-aligned bounding boxes. bounds[:, 0, :] are minimums,
        bounds[:, 1, :] are maximums.
    leaf_size : int, default=4
        Maximum primitives per leaf.
    build_method : str, default="sah"
        Build strategy: "sah" (surface area heuristic) or "median".

    Returns
    -------
    TensorDict
        Tree structure with '_type' = 'bounding_volume_hierarchy'.
    """
```

```python
def octree(
    points: Tensor,  # (n, 3)
    *,
    max_depth: int = 8,
    leaf_size: int = 10,
    bounds: Optional[Tensor] = None,
) -> TensorDict:
    """Build an octree over 3D points.

    Parameters
    ----------
    points : Tensor, shape (n, 3)
        3D points to index.
    max_depth : int, default=8
        Maximum tree depth.
    leaf_size : int, default=10
        Maximum points per leaf.
    bounds : Tensor, shape (2, 3), optional
        Custom bounding box (min, max). Computed from points if None.

    Returns
    -------
    TensorDict
        Tree structure with '_type' = 'octree'.
    """
```

### Query Functions (unified)

Query functions dispatch based on the `'_type'` field in the TensorDict:

```python
def k_nearest_neighbors(
    tree: TensorDict,
    queries: Tensor,  # (m, d)
    k: int,
    *,
    p: float = 2.0,
) -> tuple[Tensor, Tensor]:
    """Find k nearest neighbors for each query point.

    Parameters
    ----------
    tree : TensorDict
        Spatial index built by kd_tree(), bounding_volume_hierarchy(), or octree().
    queries : Tensor, shape (m, d)
        Query points.
    k : int
        Number of neighbors to find.
    p : float, default=2.0
        Minkowski p-norm (2.0 = Euclidean, 1.0 = Manhattan).

    Returns
    -------
    indices : Tensor, shape (m, k)
        Indices of k nearest neighbors per query.
    distances : Tensor, shape (m, k)
        Distances to k nearest neighbors per query.
        Supports gradients w.r.t. query points.
    """
```

```python
def radius_query(
    tree: TensorDict,
    queries: Tensor,  # (m, d)
    radius: float,
    *,
    p: float = 2.0,
) -> TensorDict:
    """Find all points within radius of each query.

    Parameters
    ----------
    tree : TensorDict
        Spatial index built by kd_tree(), bounding_volume_hierarchy(), or octree().
    queries : Tensor, shape (m, d)
        Query points.
    radius : float
        Search radius.
    p : float, default=2.0
        Minkowski p-norm.

    Returns
    -------
    TensorDict
        'indices': NestedTensor, shape (m, *) — neighbor indices
        'distances': NestedTensor, shape (m, *) — neighbor distances
        'counts': Tensor, shape (m,) — number of neighbors per query
    """
```

```python
def ray_query(
    tree: TensorDict,
    origins: Tensor,     # (m, 3)
    directions: Tensor,  # (m, 3)
    *,
    t_min: float = 0.0,
    t_max: float = float('inf'),
) -> TensorDict:
    """Find primitives intersected by each ray (BVH only).

    Parameters
    ----------
    tree : TensorDict
        BVH built by bounding_volume_hierarchy(). Raises TypeError for other tree types.
    origins : Tensor, shape (m, 3)
        Ray origins.
    directions : Tensor, shape (m, 3)
        Ray directions (need not be normalized).
    t_min : float, default=0.0
        Minimum ray parameter.
    t_max : float, default=inf
        Maximum ray parameter.

    Returns
    -------
    TensorDict
        'indices': NestedTensor, shape (m, *) — primitive indices hit
        'counts': Tensor, shape (m,) — number of hits per ray
        't_near': NestedTensor, shape (m, *) — entry distances
        't_far': NestedTensor, shape (m, *) — exit distances
    """
```

---

## TensorDict Structures

### k-d Tree

```python
TensorDict({
    '_type': 'kd_tree',
    'points': Tensor,       # (n, d) original points
    'split_dim': Tensor,    # (n_nodes,) split dimension per node
    'split_val': Tensor,    # (n_nodes,) split value per node
    'left': Tensor,         # (n_nodes,) left child index (-1 for leaf)
    'right': Tensor,        # (n_nodes,) right child index (-1 for leaf)
    'indices': Tensor,      # (n,) point indices in leaf order
    'leaf_starts': Tensor,  # (n_leaves,) start index in 'indices'
    'leaf_counts': Tensor,  # (n_leaves,) point count per leaf
})
```

### Bounding Volume Hierarchy

```python
TensorDict({
    '_type': 'bounding_volume_hierarchy',
    'bounds': Tensor,            # (n, 2, d) original primitive bounds
    'node_bounds': Tensor,       # (n_nodes, 2, d) node bounding boxes
    'left': Tensor,              # (n_nodes,) left child index
    'right': Tensor,             # (n_nodes,) right child index
    'is_leaf': Tensor,           # (n_nodes,) boolean leaf flag
    'primitive_indices': Tensor, # (n,) primitive ordering
    'leaf_starts': Tensor,       # (n_leaves,) start index in primitive_indices
    'leaf_counts': Tensor,       # (n_leaves,) primitive count per leaf
})
```

### Octree

```python
TensorDict({
    '_type': 'octree',
    'points': Tensor,        # (n, 3) original points
    'bounds': Tensor,        # (2, 3) root bounding box (min, max)
    'node_centers': Tensor,  # (n_nodes, 3) center of each node
    'node_sizes': Tensor,    # (n_nodes,) half-size of each node
    'children': Tensor,      # (n_nodes, 8) child indices per octant (-1 if empty)
    'depth': Tensor,         # (n_nodes,) depth of each node
    'is_leaf': Tensor,       # (n_nodes,) boolean leaf flag
    'indices': Tensor,       # (n,) point indices in leaf order
    'leaf_starts': Tensor,   # (n_leaves,) start index in 'indices'
    'leaf_counts': Tensor,   # (n_leaves,) point count per leaf
})
```

---

## Module Structure

```
torchscience/space_partitioning/
├── __init__.py                      # Public API exports
├── _kd_tree.py                      # kd_tree() build function
├── _bounding_volume_hierarchy.py    # bounding_volume_hierarchy() build function
├── _octree.py                       # octree() build function
├── _k_nearest_neighbors.py          # k_nearest_neighbors()
├── _radius_query.py                 # radius_query()
└── _ray_query.py                    # ray_query()
```

**`__init__.py` exports:**
```python
from ._kd_tree import kd_tree
from ._bounding_volume_hierarchy import bounding_volume_hierarchy
from ._octree import octree
from ._k_nearest_neighbors import k_nearest_neighbors
from ._radius_query import radius_query
from ._ray_query import ray_query

__all__ = [
    "kd_tree",
    "bounding_volume_hierarchy",
    "octree",
    "k_nearest_neighbors",
    "radius_query",
    "ray_query",
]
```

---

## Implementation Notes

### Gradient Support

- `k_nearest_neighbors()` and `radius_query()` return distances that support autograd
- Gradients flow through distance computation to query points
- Indices are discrete and non-differentiable
- Tree structure is built without gradient tracking

### Distance Computation

Reuse `torchscience.distance.minkowski_distance` internals or implement inline:

```python
# Minkowski distance for neighbor search
diff = query_point - tree_points  # (k, d)
if p == 2.0:
    dist = (diff ** 2).sum(-1).sqrt()
elif p == 1.0:
    dist = diff.abs().sum(-1)
elif p == float('inf'):
    dist = diff.abs().max(-1).values
else:
    dist = (diff.abs() ** p).sum(-1) ** (1/p)
```

### Build Algorithms

| Structure | Algorithm |
|-----------|-----------|
| k-d tree | Median split, cycle through dimensions |
| Bounding volume hierarchy | Surface area heuristic (SAH) or median split |
| Octree | Recursive octant subdivision |

### Future CUDA Support

When adding CUDA:
1. Query kernels first (build can stay on CPU)
2. Use warp-level primitives for tree traversal
3. Consider stackless traversal for bounding volume hierarchy

---

## Implementation Roadmap

### Phase 1: k-d Tree (MVP)

1. `kd_tree()` build function
2. `k_nearest_neighbors()` for k-NN
3. `radius_query()` with nested tensor output
4. Tests and documentation

### Phase 2: Bounding Volume Hierarchy

1. `bounding_volume_hierarchy()` build function with SAH
2. `ray_query()` for ray-primitive intersection
3. Extend `k_nearest_neighbors()`/`radius_query()` to BVH (box centers)
4. Tests and documentation

### Phase 3: Octree

1. `octree()` build function
2. Extend `k_nearest_neighbors()`/`radius_query()` to octree
3. Tests and documentation

### Future Phases

- CUDA query kernels
- Ball tree (arbitrary metrics)
- Spatial hashing (dynamic scenes)
- Quadtree (2D variant of octree)

---

## References

- Bentley, J. L. (1975). "Multidimensional binary search trees used for associative searching"
- Wald, I. (2007). "On fast Construction of SAH-based Bounding Volume Hierarchies"
- Meagher, D. (1982). "Geometric modeling using octree encoding"
