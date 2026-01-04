from __future__ import annotations

from ._polynomial import Polynomial


def polynomial_negate(p: Polynomial) -> Polynomial:
    """Negate polynomial.

    Returns
    -------
    Polynomial
        Negated polynomial -p.
    """
    return Polynomial(coeffs=-p.coeffs)
