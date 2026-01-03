"""Geometric operations and queries."""

from ._closest_point import ClosestPoint, closest_point
from ._ray_intersect import RayHit, ray_intersect
from ._ray_occluded import ray_occluded

__all__ = [
    "ClosestPoint",
    "RayHit",
    "closest_point",
    "ray_intersect",
    "ray_occluded",
]
