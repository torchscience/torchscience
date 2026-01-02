"""Differentiable spline interpolation for PyTorch tensors.

This module provides cubic splines and B-splines with full autograd support.

Convenience Functions
---------------------
cubic_spline
    Create a cubic spline interpolator from data (fit + callable).
b_spline
    Create a B-spline approximation from data (fit + callable).

Cubic Splines
-------------
cubic_spline_fit
    Fit a cubic spline to data points.
cubic_spline_evaluate
    Evaluate a cubic spline at query points.
cubic_spline_derivative
    Compute derivatives of a cubic spline.
cubic_spline_integral
    Compute definite integral of a cubic spline.

B-Splines
---------
b_spline_fit
    Fit a B-spline to data points.
b_spline_evaluate
    Evaluate a B-spline at query points.
b_spline_derivative
    Compute derivatives of a B-spline.
b_spline_basis
    Evaluate B-spline basis functions.

Data Types
----------
CubicSpline
    Piecewise cubic polynomial interpolant.
BSpline
    B-spline curve.

Exceptions
----------
SplineError
    Base exception for spline operations.
ExtrapolationError
    Query point outside spline domain.
KnotError
    Invalid knot vector.
DegreeError
    Invalid degree for given knots.
"""

from typing import Callable, Optional

import torch

from torchscience.spline._b_spline import (
    BSpline,
    b_spline_basis,
    b_spline_derivative,
    b_spline_evaluate,
    b_spline_fit,
)
from torchscience.spline._cubic_spline import (
    CubicSpline,
    cubic_spline_derivative,
    cubic_spline_evaluate,
    cubic_spline_fit,
    cubic_spline_integral,
)
from torchscience.spline._exceptions import (
    DegreeError,
    ExtrapolationError,
    KnotError,
    SplineError,
)


def cubic_spline(
    x: torch.Tensor,
    y: torch.Tensor,
    boundary: str = "not_a_knot",
    extrapolate: str = "error",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a cubic spline interpolator from data.

    This is a convenience function that fits a cubic spline and returns
    a callable that evaluates it.

    Parameters
    ----------
    x : Tensor
        Data x-coordinates. Must be strictly monotonically increasing.
    y : Tensor
        Data y-values. Shape must be compatible with x.
    boundary : str, optional
        Boundary condition type. One of:

        - ``"natural"``: Zero second derivative at endpoints.
        - ``"clamped"``: Specified first derivative at endpoints.
        - ``"not_a_knot"``: Third derivative continuity at second and
          second-to-last knots (default).
        - ``"periodic"``: Periodic boundary conditions.

    extrapolate : str, optional
        How to handle out-of-domain queries. One of:

        - ``"error"``: Raise ExtrapolationError (default).
        - ``"clamp"``: Clamp to boundary values.
        - ``"extend"``: Extrapolate using boundary polynomial.

    Returns
    -------
    spline : Callable[[Tensor], Tensor]
        Function that evaluates the spline at given points.

    Examples
    --------
    >>> import torch
    >>> x = torch.linspace(0, 1, 10)
    >>> y = torch.sin(x * 2 * torch.pi)
    >>> f = cubic_spline(x, y)
    >>> f(torch.tensor([0.5]))  # Evaluate at x=0.5
    """
    fitted = cubic_spline_fit(x, y, boundary=boundary, extrapolate=extrapolate)
    return lambda t: cubic_spline_evaluate(fitted, t)


def b_spline(
    x: torch.Tensor,
    y: torch.Tensor,
    degree: int = 3,
    n_knots: Optional[int] = None,
    extrapolate: str = "error",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a B-spline approximation from data.

    This is a convenience function that fits a B-spline and returns
    a callable that evaluates it.

    Parameters
    ----------
    x : Tensor
        Data x-coordinates. Must be strictly monotonically increasing.
    y : Tensor
        Data y-values. Shape must be compatible with x.
    degree : int, optional
        Spline degree. Default is 3 (cubic).
    n_knots : int, optional
        Number of interior knots. If not specified, a reasonable default
        is computed based on the data size.
    extrapolate : str, optional
        How to handle out-of-domain queries. One of:

        - ``"error"``: Raise ExtrapolationError (default).
        - ``"clamp"``: Clamp to boundary values.
        - ``"extend"``: Extrapolate using boundary polynomial.

    Returns
    -------
    spline : Callable[[Tensor], Tensor]
        Function that evaluates the spline at given points.

    Examples
    --------
    >>> import torch
    >>> x = torch.linspace(0, 1, 20)
    >>> y = torch.sin(x * 2 * torch.pi)
    >>> f = b_spline(x, y, degree=3, n_knots=5)
    >>> f(torch.tensor([0.5]))  # Evaluate at x=0.5
    """
    fitted = b_spline_fit(
        x, y, degree=degree, n_knots=n_knots, extrapolate=extrapolate
    )
    return lambda t: b_spline_evaluate(fitted, t)


__all__ = [
    # Convenience functions
    "cubic_spline",
    "b_spline",
    # Data types
    "BSpline",
    "CubicSpline",
    # Cubic spline functions
    "cubic_spline_derivative",
    "cubic_spline_evaluate",
    "cubic_spline_fit",
    "cubic_spline_integral",
    # B-spline functions
    "b_spline_basis",
    "b_spline_derivative",
    "b_spline_evaluate",
    "b_spline_fit",
    # Exceptions
    "SplineError",
    "ExtrapolationError",
    "KnotError",
    "DegreeError",
]
