import torch
from torch import Tensor


def chebyshev_polynomial_v(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Chebyshev polynomial of the third kind.

    Computes the Chebyshev polynomial of the third kind V_n(x).

    .. math::

        V_n(x) = \frac{\cos((n + \tfrac{1}{2})\theta)}{\cos(\theta/2)}

    where :math:`\theta = \arccos(x)`.

    Parameters
    ----------
    n : Tensor
        Degree of the polynomial. Should be non-negative.
    x : Tensor
        Input tensor. Broadcasting with n is supported.

    Returns
    -------
    Tensor
        The Chebyshev polynomial V_n(x) evaluated at the input values.

    See Also
    --------
    chebyshev_polynomial_t : Chebyshev polynomial of the first kind
    chebyshev_polynomial_u : Chebyshev polynomial of the second kind
    chebyshev_polynomial_w : Chebyshev polynomial of the fourth kind
    """
    return torch.ops.torchscience.chebyshev_polynomial_v(x, n)
