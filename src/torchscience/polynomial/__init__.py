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
>>> import torch
>>> from torchscience.polynomial import polynomial, polynomial_evaluate

>>> # Create polynomial 1 + 2x + 3x^2
>>> p = polynomial(torch.tensor([1.0, 2.0, 3.0]))

>>> # Evaluate at points
>>> x = torch.tensor([0.0, 1.0, 2.0])
>>> polynomial_evaluate(p, x)
tensor([ 1.,  6., 17.])

>>> # Operator overloading
>>> p(x)  # Same as polynomial_evaluate(p, x)
tensor([ 1.,  6., 17.])
"""

from torchscience.polynomial._composition import polynomial_compose
from torchscience.polynomial._division import (
    polynomial_div,
    polynomial_divmod,
    polynomial_mod,
)
from torchscience.polynomial._exceptions import DegreeError, PolynomialError
from torchscience.polynomial._polynomial import (
    Polynomial,
    polynomial,
    polynomial_add,
    polynomial_antiderivative,
    polynomial_degree,
    polynomial_derivative,
    polynomial_evaluate,
    polynomial_integral,
    polynomial_multiply,
    polynomial_negate,
    polynomial_pow,
    polynomial_scale,
    polynomial_subtract,
)
from torchscience.polynomial._roots import (
    polynomial_equal,
    polynomial_from_roots,
    polynomial_roots,
    polynomial_trim,
)

__all__ = [
    # Data types
    "Polynomial",
    # Constructors
    "polynomial",
    "polynomial_from_roots",
    # Arithmetic
    "polynomial_add",
    "polynomial_subtract",
    "polynomial_multiply",
    "polynomial_scale",
    "polynomial_negate",
    "polynomial_pow",
    # Division
    "polynomial_divmod",
    "polynomial_div",
    "polynomial_mod",
    # Composition
    "polynomial_compose",
    # Evaluation and calculus
    "polynomial_evaluate",
    "polynomial_derivative",
    "polynomial_antiderivative",
    "polynomial_integral",
    # Root finding
    "polynomial_roots",
    # Utilities
    "polynomial_degree",
    "polynomial_trim",
    "polynomial_equal",
    # Exceptions
    "PolynomialError",
    "DegreeError",
]
