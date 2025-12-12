from torch import Tensor

import torchscience.ops.torchscience


def hermite_polynomial_he(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Probabilist's Hermite polynomial.

    The probabilist's Hermite polynomial :math:`He_n(x)` is related to
    the physicist's Hermite polynomial by:

    .. math::
        He_n(x) = 2^{-n/2} H_n(x / \sqrt{2})

    The polynomials satisfy the recurrence relation:

    .. math::
        He_{n+1}(x) = x \cdot He_n(x) - n \cdot He_{n-1}(x)

    with :math:`He_0(x) = 1` and :math:`He_1(x) = x`.

    Parameters
    ----------
    n : Tensor
        Degree of the polynomial (non-negative integer).
    x : Tensor
        Input value.

    Returns
    -------
    Tensor
        Value of the probabilist's Hermite polynomial He_n(x).
    """
    return torchscience.ops.torchscience._hermite_polynomial_he(n, x)
