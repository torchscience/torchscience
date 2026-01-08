"""Legendre polynomial series operations."""

from ._legendre_polynomial_p import (
    LegendrePolynomialP,
    legendre_polynomial_p,
)
from ._legendre_polynomial_p_add import (
    legendre_polynomial_p_add,
)
from ._legendre_polynomial_p_evaluate import (
    legendre_polynomial_p_evaluate,
)
from ._legendre_polynomial_p_negate import (
    legendre_polynomial_p_negate,
)
from ._legendre_polynomial_p_scale import (
    legendre_polynomial_p_scale,
)
from ._legendre_polynomial_p_subtract import (
    legendre_polynomial_p_subtract,
)

__all__ = [
    "LegendrePolynomialP",
    "legendre_polynomial_p",
    "legendre_polynomial_p_add",
    "legendre_polynomial_p_evaluate",
    "legendre_polynomial_p_negate",
    "legendre_polynomial_p_scale",
    "legendre_polynomial_p_subtract",
]
