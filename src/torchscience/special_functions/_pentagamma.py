import torch
from torch import Tensor


def pentagamma(z: Tensor) -> Tensor:
    r"""
    Pentagamma function.

    Computes the third derivative of the digamma function, also known as
    the fourth logarithmic derivative of the gamma function:

    .. math::

        \psi'''(z) = \frac{d^3}{dz^3} \psi(z) = \frac{d^4}{dz^4} \ln \Gamma(z)

    This is equivalent to ``polygamma(3, z)``.

    Parameters
    ----------
    z : Tensor
        Input tensor. Poles at non-positive integers.

    Returns
    -------
    Tensor
        The pentagamma function evaluated at z.

    See Also
    --------
    digamma : First logarithmic derivative of gamma
    trigamma : Second logarithmic derivative of gamma
    tetragamma : Third logarithmic derivative of gamma
    polygamma : General nth logarithmic derivative of gamma
    """
    return torch.ops.torchscience.pentagamma(z)
