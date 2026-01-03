"""Differentiation module: finite difference operators for numerical differentiation."""

from torchscience.differentiation._apply import apply_stencil
from torchscience.differentiation._construction import (
    finite_difference_stencil,
    fornberg_weights,
)
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
    "apply_stencil",
    "finite_difference_stencil",
    "fornberg_weights",
]
