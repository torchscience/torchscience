"""Differentiation module: finite difference operators for numerical differentiation."""

from torchscience.differentiation._exceptions import (
    BoundaryError,
    DifferentiationError,
    StencilError,
)
from torchscience.differentiation._stencil import FiniteDifferenceStencil

__all__ = [
    "BoundaryError",
    "DifferentiationError",
    "FiniteDifferenceStencil",
    "StencilError",
]
