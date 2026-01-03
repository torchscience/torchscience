"""Ray-triangle intersection using BVH."""

from __future__ import annotations

import torch
from tensordict import tensorclass
from torch import Tensor

from torchscience.space_partitioning import BoundingVolumeHierarchy


@tensorclass
class RayHit:
    """Ray intersection results.

    Attributes
    ----------
    t : Tensor
        Ray parameter (distance), shape (*,). inf for miss.
    hit : Tensor
        Whether ray hit, shape (*,). bool dtype.
    geometry_id : Tensor
        Which mesh was hit, shape (*,). -1 for miss.
    primitive_id : Tensor
        Which triangle (global index), shape (*,). -1 for miss.
    u : Tensor
        Barycentric u coordinate, shape (*,).
    v : Tensor
        Barycentric v coordinate, shape (*,).
    """

    t: Tensor
    hit: Tensor
    geometry_id: Tensor
    primitive_id: Tensor
    u: Tensor
    v: Tensor


def ray_intersect(
    bvh: BoundingVolumeHierarchy,
    origins: Tensor,
    directions: Tensor,
) -> RayHit:
    """Find ray-triangle intersections.

    Parameters
    ----------
    bvh : BoundingVolumeHierarchy
        BVH structure to query.
    origins : Tensor, shape (..., 3)
        Ray origin points.
    directions : Tensor, shape (..., 3)
        Ray direction vectors (need not be normalized).

    Returns
    -------
    RayHit
        Intersection results with shape (...,).
    """
    if origins.shape[-1] != 3:
        raise ValueError(
            f"origins must have shape (..., 3), got {origins.shape}"
        )
    if directions.shape[-1] != 3:
        raise ValueError(
            f"directions must have shape (..., 3), got {directions.shape}"
        )

    # Broadcast origins and directions
    origins, directions = torch.broadcast_tensors(origins, directions)

    # Get scene handle
    scene_handle = bvh._scene_handles.item()

    # Call C++ kernel
    t, hit, geometry_id, primitive_id, u, v = (
        torch.ops.torchscience.bvh_ray_intersect(
            scene_handle,
            origins.contiguous(),
            directions.contiguous(),
        )
    )

    return RayHit(
        t=t,
        hit=hit,
        geometry_id=geometry_id,
        primitive_id=primitive_id,
        u=u,
        v=v,
        batch_size=list(origins.shape[:-1]),
    )
