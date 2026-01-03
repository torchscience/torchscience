"""Closest point queries using BVH."""

from __future__ import annotations

import torch
from tensordict import tensorclass
from torch import Tensor

from torchscience.space_partitioning import BoundingVolumeHierarchy


@tensorclass
class ClosestPoint:
    """Closest point query results.

    Attributes
    ----------
    point : Tensor
        Closest point on surface, shape (*, 3).
    distance : Tensor
        Distance to closest point, shape (*,).
    geometry_id : Tensor
        Which mesh, shape (*,).
    primitive_id : Tensor
        Which triangle (global index), shape (*,).
    u : Tensor
        Barycentric u coordinate, shape (*,).
    v : Tensor
        Barycentric v coordinate, shape (*,).
    """

    point: Tensor
    distance: Tensor
    geometry_id: Tensor
    primitive_id: Tensor
    u: Tensor
    v: Tensor


def closest_point(
    bvh: BoundingVolumeHierarchy,
    query_points: Tensor,
) -> ClosestPoint:
    """Find closest points on mesh surface.

    Parameters
    ----------
    bvh : BoundingVolumeHierarchy
        BVH structure to query.
    query_points : Tensor, shape (..., 3)
        Points to find closest surface points for.

    Returns
    -------
    ClosestPoint
        Query results with shape (...,).
    """
    if query_points.shape[-1] != 3:
        raise ValueError(
            f"query_points must have shape (..., 3), got {query_points.shape}"
        )

    scene_handle = bvh._scene_handles.item()

    point, distance, geometry_id, primitive_id, u, v = (
        torch.ops.torchscience.bvh_closest_point(
            scene_handle,
            query_points.contiguous(),
        )
    )

    return ClosestPoint(
        point=point,
        distance=distance,
        geometry_id=geometry_id,
        primitive_id=primitive_id,
        u=u,
        v=v,
        batch_size=list(query_points.shape[:-1]),
    )
