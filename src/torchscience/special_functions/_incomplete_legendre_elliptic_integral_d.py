import torch
from torch import Tensor

import torchscience

__all__ = ["incomplete_legendre_elliptic_integral_d"]


def incomplete_legendre_elliptic_integral_d(phi: Tensor, k: Tensor) -> Tensor:
    r"""
    Incomplete Legendre elliptic integral :math:`D(\phi, k)`.

    .. math::
        D(\phi, k) = \frac{F(\phi, k) - E(\phi, k)}{k^2}

    where :math:`F(\phi, k)` and :math:`E(\phi, k)` are the incomplete elliptic
    integrals of the first and second kind respectively.

    Args:
        phi (Tensor): The amplitude (upper limit of integration, in radians).
        k (Tensor): The elliptic modulus (must satisfy |k| < 1).

    Returns:
        Tensor: The value of the incomplete Legendre elliptic integral D.
    """
    return torchscience.ops.torchscience._incomplete_legendre_elliptic_integral_d(phi, k)
