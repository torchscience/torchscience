from torch import Tensor

import torchscience.ops.torchscience

__all__ = ["legendre_q"]


def legendre_q(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Legendre function of the second kind :math:`Q_n(x)`.

    The Legendre functions of the second kind are solutions to Legendre's
    differential equation that are singular at :math:`x = \pm 1`.

    For :math:`|x| > 1`, they can be expressed as:

    .. math::
        Q_n(x) = \frac{1}{2} P_n(x) \ln\left(\frac{x+1}{x-1}\right) - W_{n-1}(x)

    where :math:`W_{n-1}(x)` is a polynomial of degree :math:`n-1`.

    Parameters
    ----------
    n : Tensor
        Non-negative integer degree.
    x : Tensor
        Argument, with :math:`|x| > 1` for real values.

    Returns
    -------
    Tensor
        The Legendre function of the second kind :math:`Q_n(x)`.
    """
    return torchscience.ops.torchscience._legendre_q(n, x)
