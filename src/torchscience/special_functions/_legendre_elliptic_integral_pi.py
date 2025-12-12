import torch
from torch import Tensor

import torchscience

__all__ = ["legendre_elliptic_integral_pi"]


def legendre_elliptic_integral_pi(n: Tensor, phi: Tensor, k: Tensor) -> Tensor:
    r"""
    Incomplete elliptic integral of the third kind (Legendre form) :math:`\Pi(n; \varphi \mid k)`.

    .. math::
        \Pi(n; \varphi \mid k) = \int_0^{\varphi} \frac{d\theta}{(1 - n \sin^2 \theta) \sqrt{1 - k^2 \sin^2 \theta}}

    This is also known as Legendre's incomplete elliptic integral of the third kind.

    Args:
        n (Tensor): Characteristic parameter.
        phi (Tensor): Amplitude (in radians).
        k (Tensor): Elliptic modulus (must satisfy :math:`|k| \leq 1`).

    Returns:
        Tensor: The value of the incomplete elliptic integral of the third kind.
    """
    return torchscience.ops.torchscience._legendre_elliptic_integral_pi(n, phi, k)
