import torch
from torch import Tensor

import torchscience

__all__ = ["complete_carlson_elliptic_r_f"]


def complete_carlson_elliptic_r_f(x: Tensor, y: Tensor) -> Tensor:
    r"""
    Complete Carlson's elliptic integral :math:`R_F(0, x, y)`.

    .. math::
        R_F(0, x, y) = \frac{1}{2} \int_0^{\infty} \frac{dt}{\sqrt{t(t + x)(t + y)}}

    This is the complete case of Carlson's elliptic integral of the first kind
    where the first argument is zero.

    Args:
        x (Tensor): Input tensor for x (must be non-negative).
        y (Tensor): Input tensor for y (must be positive).

    Returns:
        Tensor: The value of the complete Carlson elliptic integral R_F(0, x, y).
    """
    return torchscience.ops.torchscience._complete_carlson_elliptic_r_f(x, y)
