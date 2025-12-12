import torch
from torch import Tensor

import torchscience

__all__ = ["chebyshev_polynomial_u"]


def chebyshev_polynomial_u(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Chebyshev polynomial of the second kind :math:`U_n(x)`.

    .. math::
        U_n(x) = \frac{\sin((n+1) \arccos(x))}{\sin(\arccos(x))} \quad \text{for } |x| \leq 1

    The Chebyshev polynomials of the second kind satisfy the recurrence relation:

    .. math::
        U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)

    with :math:`U_0(x) = 1` and :math:`U_1(x) = 2x`.

    Args:
        n (Tensor): Degree of the polynomial (non-negative integer).
        x (Tensor): Argument of the polynomial.

    Returns:
        Tensor: The Chebyshev polynomial of the second kind evaluated at x.
    """
    return torchscience.ops.torchscience._chebyshev_polynomial_u(n, x)
