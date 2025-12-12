import torch
from torch import Tensor

import torchscience

__all__ = ["complete_elliptic_integral_e"]


def complete_elliptic_integral_e(input: Tensor) -> Tensor:
    r"""
    Complete elliptic integral of the second kind :math:`E(k)`.

    .. math::
        E(k) = \int_0^{\pi/2} \sqrt{1 - k^2 \sin^2 t} \, dt

    This integral represents the arc length of an ellipse with semi-major
    axis 1 and eccentricity k.

    Args:
        input (Tensor): Input tensor (elliptic modulus k, must satisfy |k| ≤ 1).

    Returns:
        Tensor: The value of the complete elliptic integral of the second kind.
    """
    return torchscience.ops.torchscience._complete_elliptic_integral_e(input)
