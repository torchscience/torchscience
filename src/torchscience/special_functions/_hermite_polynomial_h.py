from torch import Tensor

import torchscience.ops.torchscience


def hermite_polynomial_h(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Physicist's Hermite polynomial.

    The physicist's Hermite polynomial :math:`H_n(x)` satisfies the
    recurrence relation:

    .. math::
        H_{n+1}(x) = 2x \cdot H_n(x) - 2n \cdot H_{n-1}(x)

    with :math:`H_0(x) = 1` and :math:`H_1(x) = 2x`.

    Related to the probabilist's Hermite polynomial by:

    .. math::
        H_n(x) = 2^{n/2} He_n(x \sqrt{2})

    Parameters
    ----------
    n : Tensor
        Degree of the polynomial (non-negative integer).
    x : Tensor
        Input value.

    Returns
    -------
    Tensor
        Value of the physicist's Hermite polynomial H_n(x).
    """
    return torchscience.ops.torchscience._hermite_polynomial_h(n, x)
