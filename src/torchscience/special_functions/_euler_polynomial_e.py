from torch import Tensor

import torchscience.ops.torchscience

__all__ = ["euler_polynomial_e"]


def euler_polynomial_e(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Euler polynomial :math:`E_n(x)`.

    The Euler polynomials are defined by the generating function:

    .. math::
        \frac{2e^{xt}}{e^t + 1} = \sum_{n=0}^{\infty} E_n(x) \frac{t^n}{n!}

    Special cases:

    - :math:`E_0(x) = 1`
    - :math:`E_1(x) = x - \frac{1}{2}`
    - :math:`E_2(x) = x^2 - x`

    The derivative satisfies :math:`\frac{d}{dx} E_n(x) = n E_{n-1}(x)`.

    Parameters
    ----------
    n : Tensor
        Non-negative integer degree of the polynomial.
    x : Tensor
        Argument at which to evaluate the polynomial.

    Returns
    -------
    Tensor
        The Euler polynomial :math:`E_n(x)`.
    """
    return torchscience.ops.torchscience._euler_polynomial_e(n, x)
