import torch
from torch import Tensor

import torchscience

__all__ = ["incomplete_elliptic_integral_f"]


def incomplete_elliptic_integral_f(phi: Tensor, k: Tensor) -> Tensor:
    r"""
    Incomplete elliptic integral of the first kind :math:`F(\phi, k)`.

    .. math::
        F(\phi, k) = \int_0^{\phi} \frac{dt}{\sqrt{1 - k^2 \sin^2 t}}

    This integral appears in the solution of the simple pendulum problem
    and many other problems in physics and engineering.

    Args:
        phi (Tensor): The amplitude (upper limit of integration, in radians).
        k (Tensor): The elliptic modulus (must satisfy |k| < 1).

    Returns:
        Tensor: The value of the incomplete elliptic integral of the first kind.
    """
    return torchscience.ops.torchscience._incomplete_elliptic_integral_f(phi, k)
