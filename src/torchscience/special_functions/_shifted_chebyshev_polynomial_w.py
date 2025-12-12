from torch import Tensor

import torchscience.ops.torchscience


def shifted_chebyshev_polynomial_w(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Shifted Chebyshev polynomial of the fourth kind.

    .. math::
        W^*_n(x) = W_n(2x - 1)

    where :math:`W_n` is the Chebyshev polynomial of the fourth kind.
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
        Value of the shifted Chebyshev polynomial W*_n(x).
    """
    return torchscience.ops.torchscience._shifted_chebyshev_polynomial_w(n, x)
