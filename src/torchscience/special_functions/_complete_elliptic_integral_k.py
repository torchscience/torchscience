import torch
from torch import Tensor

import torchscience

__all__ = ["complete_elliptic_integral_k"]


def complete_elliptic_integral_k(input: Tensor) -> Tensor:
    r"""
    Complete elliptic integral of the first kind :math:`K(k)`.

    .. math::
        K(k) = \int_0^{\pi/2} \frac{dt}{\sqrt{1 - k^2 \sin^2 t}}

    This integral is related to the arc length of an ellipse and appears
    in many physics applications including pendulum motion and electromagnetic theory.

    Args:
        input (Tensor): Input tensor (elliptic modulus k, must satisfy |k| < 1).

    Returns:
        Tensor: The value of the complete elliptic integral of the first kind.
    """
    return torchscience.ops.torchscience._complete_elliptic_integral_k(input)
