import torch
from torch import Tensor

import torchscience

__all__ = ["chebyshev_polynomial_t"]


def chebyshev_polynomial_t(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Chebyshev polynomial of the first kind :math:`T_n(x)`.

    .. math::
        T_n(x) = \cos(n \arccos(x)) \quad \text{for } |x| \leq 1

    The Chebyshev polynomials satisfy the recurrence relation:

    .. math::
        T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)

    with :math:`T_0(x) = 1` and :math:`T_1(x) = x`.

    Args:
        n (Tensor): Degree of the polynomial (non-negative integer).
        x (Tensor): Argument of the polynomial.

    Returns:
        Tensor: The Chebyshev polynomial of the first kind evaluated at x.
    """
    return torchscience.ops.torchscience._chebyshev_polynomial_t(n, x)
