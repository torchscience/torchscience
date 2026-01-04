"""
Numerical integration (quadrature) module.

Sample-based integration (operates on pre-computed values):
    trapezoid, cumulative_trapezoid, simpson, cumulative_simpson

Function-based integration (evaluates callable):
    fixed_quad, quad, quad_info, dblquad

Quadrature rule classes:
    GaussLegendre, GaussKronrod
"""

from torchscience.integration.quadrature._exceptions import (
    IntegrationError,
    QuadratureWarning,
)
from torchscience.integration.quadrature._trapezoid import (
    cumulative_trapezoid,
    trapezoid,
)

__all__ = [
    "trapezoid",
    "cumulative_trapezoid",
    "QuadratureWarning",
    "IntegrationError",
]
