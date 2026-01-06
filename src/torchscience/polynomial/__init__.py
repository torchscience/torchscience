"""Differentiable polynomial arithmetic for PyTorch tensors.

This module provides polynomial arithmetic with full autograd support.
Polynomials use ascending coefficient order (NumPy convention) with
batch dimensions first.

Constructors
------------
polynomial
    Create polynomial from coefficients.
polynomial_from_roots
    Construct monic polynomial from its roots.

Arithmetic
----------
polynomial_add
    Add two polynomials.
polynomial_subtract
    Subtract polynomials.
polynomial_multiply
    Multiply polynomials.
polynomial_scale
    Multiply polynomial by scalar.
polynomial_negate
    Negate polynomial.
polynomial_pow
    Raise polynomial to non-negative integer power.

Division
--------
polynomial_divmod
    Divide polynomials, returning quotient and remainder.
polynomial_div
    Quotient of polynomial division.
polynomial_mod
    Remainder of polynomial division.

Composition
-----------
polynomial_compose
    Compute p(q(x)).

Evaluation and Calculus
-----------------------
polynomial_evaluate
    Evaluate polynomial at points.
polynomial_derivative
    Compute derivative.
polynomial_antiderivative
    Compute antiderivative.
polynomial_integral
    Compute definite integral.

Fitting
-------
polynomial_fit
    Fit polynomial to data via least squares.
polynomial_vandermonde
    Construct Vandermonde matrix.

Root Finding
------------
polynomial_roots
    Find roots via companion matrix eigenvalues.

Utilities
---------
polynomial_degree
    Return degree of polynomial.
polynomial_trim
    Remove trailing near-zero coefficients.
polynomial_equal
    Check polynomial equality within tolerance.

Data Types
----------
Polynomial
    Polynomial in power basis with ascending coefficients.

Exceptions
----------
PolynomialError
    Base exception for polynomial operations.
DegreeError
    Invalid degree for operation.

Examples
--------
>>> from torchscience.polynomial._polynomial_evaluate import polynomial_evaluate import torch
>>> from torchscience.polynomial import polynomial

>>> # Create polynomial 1 + 2x + 3x^2
>>> p = polynomial(torch.tensor([1.0, 2.0, 3.0]))

>>> # Evaluate at points
>>> x = torch.tensor([0.0, 1.0, 2.0])
>>> polynomial_evaluate(p, x)
tensor([ 1.,  6., 17.])

>>> # Operator overloading
>>> p(x)  # Same as polynomial_evaluate(p, x)
tensor([ 1.,  6., 17.])

>>> # Division
>>> q = polynomial(torch.tensor([1.0, 1.0]))  # 1 + x
>>> p // q  # Quotient
>>> p % q   # Remainder

>>> # Power
>>> q ** 3  # (1 + x)^3

>>> from torchscience.polynomial._polynomial_fit import polynomial_fit # Fitting
>>> x = torch.tensor([0.0, 1.0, 2.0, 3.0])
>>> y = torch.tensor([1.0, 3.0, 5.0, 7.0])
>>> p = polynomial_fit(x, y, degree=1)  # Fits y = 1 + 2x
"""

from ._chebyshev_t import (
    ChebyshevT,
    chebyshev_t,
    chebyshev_t_add,
    chebyshev_t_antiderivative,
    chebyshev_t_companion,
    chebyshev_t_degree,
    chebyshev_t_derivative,
    chebyshev_t_equal,
    chebyshev_t_evaluate,
    chebyshev_t_fit,
    chebyshev_t_from_roots,
    chebyshev_t_integral,
    chebyshev_t_interpolate,
    chebyshev_t_multiply,
    chebyshev_t_mulx,
    chebyshev_t_negate,
    chebyshev_t_points,
    chebyshev_t_pow,
    chebyshev_t_roots,
    chebyshev_t_scale,
    chebyshev_t_subtract,
    chebyshev_t_to_polynomial,
    chebyshev_t_trim,
    chebyshev_t_vandermonde,
    chebyshev_t_weight,
    polynomial_to_chebyshev_t,
)
from ._degree_error import DegreeError
from ._polynomial import (
    Polynomial,
    polynomial,
    polynomial_add,
    polynomial_antiderivative,
    polynomial_compose,
    polynomial_degree,
    polynomial_derivative,
    polynomial_div,
    polynomial_divmod,
    polynomial_equal,
    polynomial_evaluate,
    polynomial_fit,
    polynomial_from_roots,
    polynomial_integral,
    polynomial_mod,
    polynomial_multiply,
    polynomial_negate,
    polynomial_pow,
    polynomial_roots,
    polynomial_scale,
    polynomial_subtract,
    polynomial_trim,
    polynomial_vandermonde,
)
from ._polynomial_error import (
    PolynomialError,
)

__all__ = [
    "ChebyshevT",
    "chebyshev_t",
    "chebyshev_t_add",
    "chebyshev_t_antiderivative",
    "chebyshev_t_companion",
    "chebyshev_t_degree",
    "chebyshev_t_derivative",
    "chebyshev_t_equal",
    "chebyshev_t_evaluate",
    "chebyshev_t_fit",
    "chebyshev_t_from_roots",
    "chebyshev_t_integral",
    "chebyshev_t_interpolate",
    "chebyshev_t_multiply",
    "chebyshev_t_mulx",
    "chebyshev_t_negate",
    "chebyshev_t_points",
    "chebyshev_t_pow",
    "chebyshev_t_roots",
    "chebyshev_t_scale",
    "chebyshev_t_subtract",
    "chebyshev_t_to_polynomial",
    "chebyshev_t_trim",
    "chebyshev_t_vandermonde",
    "chebyshev_t_weight",
    "polynomial_to_chebyshev_t",
    "DegreeError",
    "Polynomial",
    "PolynomialError",
    "polynomial",
    "polynomial_add",
    "polynomial_antiderivative",
    "polynomial_compose",
    "polynomial_degree",
    "polynomial_derivative",
    "polynomial_div",
    "polynomial_divmod",
    "polynomial_equal",
    "polynomial_evaluate",
    "polynomial_fit",
    "polynomial_from_roots",
    "polynomial_integral",
    "polynomial_mod",
    "polynomial_multiply",
    "polynomial_negate",
    "polynomial_pow",
    "polynomial_roots",
    "polynomial_scale",
    "polynomial_subtract",
    "polynomial_trim",
    "polynomial_vandermonde",
]
