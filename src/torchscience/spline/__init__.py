"""Differentiable spline interpolation for PyTorch tensors.

This module provides cubic splines and B-splines with full autograd support.

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
cubic_spline_interpolate
    Convenience function: fit + evaluate in one call.

B-Splines
---------
b_spline
    Construct a B-spline from knots and control points.
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

from torchscience.spline._cubic_spline import (
    CubicSpline,
    cubic_spline_evaluate,
    cubic_spline_fit,
)
from torchscience.spline._exceptions import (
    DegreeError,
    ExtrapolationError,
    KnotError,
    SplineError,
)

__all__ = [
    # Data types
    "CubicSpline",
    # Cubic spline functions
    "cubic_spline_evaluate",
    "cubic_spline_fit",
    # Exceptions
    "SplineError",
    "ExtrapolationError",
    "KnotError",
    "DegreeError",
]
