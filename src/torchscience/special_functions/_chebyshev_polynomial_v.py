import torch
from torch import Tensor

import torchscience

__all__ = ["chebyshev_polynomial_v"]


def chebyshev_polynomial_v(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Chebyshev polynomial of the third kind :math:`V_n(x)`.

    .. math::
        V_n(x) = \frac{\cos((n + 1/2) \arccos(x))}{\cos(\arccos(x)/2)} \quad \text{for } |x| \leq 1

    The Chebyshev polynomials of the third kind satisfy the recurrence relation:

    .. math::
        V_{n+1}(x) = 2x V_n(x) - V_{n-1}(x)

    with :math:`V_0(x) = 1` and :math:`V_1(x) = 2x - 1`.

    Args:
        n (Tensor): Degree of the polynomial (non-negative integer).
        x (Tensor): Argument of the polynomial.

    Returns:
        Tensor: The Chebyshev polynomial of the third kind evaluated at x.
    """
    return torchscience.ops.torchscience._chebyshev_polynomial_v(n, x)
