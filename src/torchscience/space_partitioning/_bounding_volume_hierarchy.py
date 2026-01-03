"""Bounding volume hierarchy implementation using Embree."""

from __future__ import annotations

from typing import Sequence

import torch
from tensordict import tensorclass
from torch import Tensor


@tensorclass
class BoundingVolumeHierarchy:
    """Bounding volume hierarchy for triangle meshes.

    Wraps Embree RTCScene for efficient ray-triangle queries.
    Use `bounding_volume_hierarchy()` to construct instances.

    Attributes
    ----------
    vertices : Tensor
        Concatenated vertices from all meshes, shape (*, V_total, 3).
    faces : Tensor
        Concatenated faces with adjusted indices, shape (*, F_total, 3).
    mesh_offsets : Tensor
        Vertex offset per mesh for index remapping, shape (*, M + 1).
    face_offsets : Tensor
        Face offset per mesh for geometry_id lookup, shape (*, M + 1).
    _scene_handles : Tensor
        Opaque handles to Embree RTCScenes, shape (*,).

    Notes
    -----
    BVH construction is NOT differentiable (discrete structure).
    Query operations (ray_intersect, closest_point) support autograd.
    """

    vertices: Tensor
    faces: Tensor
    mesh_offsets: Tensor
    face_offsets: Tensor
    _scene_handles: Tensor


def bounding_volume_hierarchy(
    meshes: Sequence[tuple[Tensor, Tensor]],
) -> BoundingVolumeHierarchy:
    """Build a bounding volume hierarchy from triangle meshes.

    Parameters
    ----------
    meshes : Sequence[tuple[Tensor, Tensor]]
        List of (vertices, faces) tuples. Each vertices tensor has shape
        (V_i, 3) and each faces tensor has shape (F_i, 3) with vertex indices.

    Returns
    -------
    BoundingVolumeHierarchy
        BVH structure ready for ray queries.

    Examples
    --------
    >>> vertices = torch.randn(100, 3)
    >>> faces = torch.randint(0, 100, (50, 3))
    >>> bvh = bounding_volume_hierarchy([(vertices, faces)])
    """
    if not meshes:
        raise ValueError("meshes must not be empty")

    # Concatenate all vertices and faces
    all_vertices = []
    all_faces = []
    vertex_offsets = [0]
    face_offsets = [0]

    for vertices, faces in meshes:
        if vertices.dim() != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (V, 3), got {vertices.shape}")
        if faces.dim() != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must be (F, 3), got {faces.shape}")

        # Adjust face indices by current vertex offset
        adjusted_faces = faces + vertex_offsets[-1]
        all_vertices.append(vertices)
        all_faces.append(adjusted_faces)

        vertex_offsets.append(vertex_offsets[-1] + vertices.shape[0])
        face_offsets.append(face_offsets[-1] + faces.shape[0])

    concat_vertices = torch.cat(all_vertices, dim=0)
    concat_faces = torch.cat(all_faces, dim=0)
    mesh_offsets = torch.tensor(
        vertex_offsets, dtype=torch.long, device=concat_vertices.device
    )
    face_offsets_tensor = torch.tensor(
        face_offsets, dtype=torch.long, device=concat_vertices.device
    )

    # Build Embree scene via C++ kernel
    scene_handle = torch.ops.torchscience.bvh_build(
        concat_vertices.contiguous(),
        concat_faces.contiguous(),
    )

    return BoundingVolumeHierarchy(
        vertices=concat_vertices,
        faces=concat_faces,
        mesh_offsets=mesh_offsets,
        face_offsets=face_offsets_tensor,
        _scene_handles=scene_handle,
        batch_size=[],
    )
