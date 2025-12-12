import torch
from torch import Tensor

import torchscience

__all__ = ["complete_legendre_elliptic_integral_d"]


def complete_legendre_elliptic_integral_d(input: Tensor) -> Tensor:
    r"""
    Complete Legendre elliptic integral :math:`D(k)`.

    .. math::
        D(k) = \frac{K(k) - E(k)}{k^2}

    where :math:`K(k)` and :math:`E(k)` are the complete elliptic integrals of the
    first and second kind respectively.

    Args:
        input (Tensor): Input tensor (elliptic modulus k, must satisfy |k| < 1).

    Returns:
        Tensor: The value of the complete Legendre elliptic integral D.
    """
    return torchscience.ops.torchscience._complete_legendre_elliptic_integral_d(input)
