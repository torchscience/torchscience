import torch
from torch import Tensor

import torchscience

__all__ = ["chebyshev_polynomial_w"]


def chebyshev_polynomial_w(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Chebyshev polynomial of the fourth kind :math:`W_n(x)`.

    .. math::
        W_n(x) = \frac{\sin((n + 1/2) \arccos(x))}{\sin(\arccos(x)/2)} \quad \text{for } |x| \leq 1

    The Chebyshev polynomials of the fourth kind satisfy the recurrence relation:

    .. math::
        W_{n+1}(x) = 2x W_n(x) - W_{n-1}(x)

    with :math:`W_0(x) = 1` and :math:`W_1(x) = 2x + 1`.

    Args:
        n (Tensor): Degree of the polynomial (non-negative integer).
        x (Tensor): Argument of the polynomial.

    Returns:
        Tensor: The Chebyshev polynomial of the fourth kind evaluated at x.
    """
    return torchscience.ops.torchscience._chebyshev_polynomial_w(n, x)
