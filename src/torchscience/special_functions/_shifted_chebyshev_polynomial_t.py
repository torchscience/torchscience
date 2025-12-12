from torch import Tensor

import torchscience.ops.torchscience


def shifted_chebyshev_polynomial_t(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Shifted Chebyshev polynomial of the first kind.

    .. math::
        T^*_n(x) = T_n(2x - 1)

    where :math:`T_n` is the Chebyshev polynomial of the first kind.
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
        Value of the shifted Chebyshev polynomial T*_n(x).
    """
    return torchscience.ops.torchscience._shifted_chebyshev_polynomial_t(n, x)
