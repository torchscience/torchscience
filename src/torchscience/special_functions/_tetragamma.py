import torch
from torch import Tensor


def tetragamma(z: Tensor) -> Tensor:
    r"""
    Tetragamma function.

    Computes the second derivative of the digamma function, also known as
    the third logarithmic derivative of the gamma function:

    .. math::

        \psi''(z) = \frac{d^2}{dz^2} \psi(z) = \frac{d^3}{dz^3} \ln \Gamma(z)

    This is equivalent to ``polygamma(2, z)``.

    Parameters
    ----------
    z : Tensor
        Input tensor. Poles at non-positive integers.

    Returns
    -------
    Tensor
        The tetragamma function evaluated at z.

    See Also
    --------
    digamma : First logarithmic derivative of gamma
    trigamma : Second logarithmic derivative of gamma
    pentagamma : Fourth logarithmic derivative of gamma
    polygamma : General nth logarithmic derivative of gamma
    """
    return torch.ops.torchscience.tetragamma(z)
