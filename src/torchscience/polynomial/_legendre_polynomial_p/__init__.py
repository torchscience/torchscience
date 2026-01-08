"""Legendre polynomial series operations."""

from ._legendre_polynomial_p import (
    LegendrePolynomialP,
    legendre_polynomial_p,
)
from ._legendre_polynomial_p_evaluate import (
    legendre_polynomial_p_evaluate,
)

__all__ = [
    "LegendrePolynomialP",
    "legendre_polynomial_p",
    "legendre_polynomial_p_evaluate",
]
