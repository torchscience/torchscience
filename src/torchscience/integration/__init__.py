"""
Numerical integration methods.

Submodules
----------
initial_value_problem
    Solvers for initial value problems (ODEs).
quadrature
    Numerical integration (quadrature) methods.
"""

from torchscience.integration import initial_value_problem, quadrature

__all__ = [
    "initial_value_problem",
    "quadrature",
]
