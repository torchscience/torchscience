from torch import Tensor

import torchscience.ops.torchscience

__all__ = ["legendre_p"]


def legendre_p(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Legendre polynomial of the first kind :math:`P_n(x)`.

    The Legendre polynomials satisfy the recurrence relation:

    .. math::
        (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)

    with :math:`P_0(x) = 1` and :math:`P_1(x) = x`.

    They are orthogonal on :math:`[-1, 1]` with weight function :math:`w(x) = 1`:

    .. math::
        \int_{-1}^{1} P_m(x) P_n(x) dx = \frac{2}{2n+1} \delta_{mn}

    Parameters
    ----------
    n : Tensor
        Non-negative integer degree of the polynomial.
    x : Tensor
        Argument, typically in :math:`[-1, 1]`.

    Returns
    -------
    Tensor
        The Legendre polynomial :math:`P_n(x)`.
    """
    return torchscience.ops.torchscience._legendre_p(n, x)
