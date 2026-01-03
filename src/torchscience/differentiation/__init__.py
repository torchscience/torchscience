"""Differentiation module: finite difference operators for numerical differentiation."""

from torchscience.differentiation._apply import apply_stencil
from torchscience.differentiation._construction import (
    biharmonic_stencil,
    finite_difference_stencil,
    fornberg_weights,
    gradient_stencils,
    laplacian_stencil,
)
from torchscience.differentiation._exceptions import (
    BoundaryError,
    DifferentiationError,
    StencilError,
)
from torchscience.differentiation._richardson import richardson_extrapolation
from torchscience.differentiation._scalar_operators import (
    biharmonic,
    derivative,
    gradient,
    hessian,
    laplacian,
)
from torchscience.differentiation._stencil import FiniteDifferenceStencil
from torchscience.differentiation._vector_operators import (
    curl,
    divergence,
    jacobian,
)

__all__ = [
    "BoundaryError",
    "DifferentiationError",
    "FiniteDifferenceStencil",
    "StencilError",
    "apply_stencil",
    "biharmonic",
    "biharmonic_stencil",
    "curl",
    "derivative",
    "divergence",
    "finite_difference_stencil",
    "fornberg_weights",
    "gradient",
    "gradient_stencils",
    "hessian",
    "jacobian",
    "laplacian",
    "laplacian_stencil",
    "richardson_extrapolation",
]
