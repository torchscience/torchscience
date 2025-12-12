from torch import Tensor

import torchscience.ops.torchscience


def shifted_chebyshev_polynomial_v(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Shifted Chebyshev polynomial of the third kind.

    .. math::
        V^*_n(x) = V_n(2x - 1)

    where :math:`V_n` is the Chebyshev polynomial of the third kind.
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
        Value of the shifted Chebyshev polynomial V*_n(x).
    """
    return torchscience.ops.torchscience._shifted_chebyshev_polynomial_v(n, x)
