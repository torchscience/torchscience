import torch
from torch import Tensor

import torchscience

__all__ = ["complete_elliptic_integral_pi"]


def complete_elliptic_integral_pi(n: Tensor, k: Tensor) -> Tensor:
    r"""
    Complete elliptic integral of the third kind :math:`\Pi(n, k)`.

    .. math::
        \Pi(n, k) = \int_0^{\pi/2} \frac{dt}{(1 - n \sin^2 t) \sqrt{1 - k^2 \sin^2 t}}

    This integral appears in problems involving the motion of a pendulum,
    the rotation of a rigid body, and other mechanics problems.

    Args:
        n (Tensor): The characteristic (elliptic characteristic).
        k (Tensor): The elliptic modulus (must satisfy |k| < 1).

    Returns:
        Tensor: The value of the complete elliptic integral of the third kind.
    """
    return torchscience.ops.torchscience._complete_elliptic_integral_pi(n, k)
