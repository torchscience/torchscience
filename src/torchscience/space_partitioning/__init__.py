"""Spatial data structures for efficient neighbor search and range queries.

This module provides k-d trees with:
- O(log n) query performance via tree traversal
- Autograd support for query operations (build is non-differentiable)
- Batched construction for (B, N, D) point clouds
- Thread-safe concurrent queries

Note: Tree construction produces a discrete data structure and is NOT
differentiable. Query operations (k_nearest_neighbors, range_search)
support autograd through the returned distances.
"""

from ._bounding_volume_hierarchy import (
    BoundingVolumeHierarchy,
    bounding_volume_hierarchy,
)
from ._k_nearest_neighbors import k_nearest_neighbors
from ._kd_tree import KdTree, kd_tree
from ._range_search import range_search

__all__ = [
    "BoundingVolumeHierarchy",
    "KdTree",
    "bounding_volume_hierarchy",
    "k_nearest_neighbors",
    "kd_tree",
    "range_search",
]
