from torch import Tensor

import torchscience.ops.torchscience


def shifted_chebyshev_polynomial_u(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Shifted Chebyshev polynomial of the second kind.

    .. math::
        U^*_n(x) = U_n(2x - 1)

    where :math:`U_n` is the Chebyshev polynomial of the second kind.
    The shifted Chebyshev polynomial is defined on [0, 1] instead of [-1, 1].

    Parameters
    ----------
    n : Tensor
        Degree of the polynomial (non-negative integer).
    x : Tensor
        Input value, typically in [0, 1].

    Returns
    -------
    Tensor
        Value of the shifted Chebyshev polynomial U*_n(x).
    """
    return torchscience.ops.torchscience._shifted_chebyshev_polynomial_u(n, x)
