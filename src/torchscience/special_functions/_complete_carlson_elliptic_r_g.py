import torch
from torch import Tensor

import torchscience

__all__ = ["complete_carlson_elliptic_r_g"]


def complete_carlson_elliptic_r_g(x: Tensor, y: Tensor) -> Tensor:
    r"""
    Complete Carlson's elliptic integral :math:`R_G(0, x, y)`.

    .. math::
        R_G(0, x, y) = \frac{1}{4\pi} \int_0^{2\pi} \int_0^{\pi} \sqrt{x\sin^2\theta\cos^2\phi + y\sin^2\theta\sin^2\phi} \sin\theta \, d\theta \, d\phi

    This is the complete case of Carlson's elliptic integral of the second kind
    where the first argument is zero.

    Args:
        x (Tensor): Input tensor for x (must be non-negative).
        y (Tensor): Input tensor for y (must be non-negative, and at least one of x or y must be positive).

    Returns:
        Tensor: The value of the complete Carlson elliptic integral R_G(0, x, y).
    """
    return torchscience.ops.torchscience._complete_carlson_elliptic_r_g(x, y)
