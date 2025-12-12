import torch
from torch import Tensor

import torchscience

__all__ = ["carlson_elliptic_integral_r_e"]


def carlson_elliptic_integral_r_e(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    r"""
    Carlson's elliptic integral :math:`R_F(x, y, z)`.

    This is the fundamental Carlson symmetric elliptic integral:

    .. math::
        R_F(x, y, z) = \frac{1}{2} \int_0^{\infty} \frac{dt}{\sqrt{(t + x)(t + y)(t + z)}}

    Note: This function provides the R_F integral under the alternative naming
    convention r_e.

    Args:
        x (Tensor): First argument (must be non-negative).
        y (Tensor): Second argument (must be non-negative).
        z (Tensor): Third argument (must be non-negative, at most one can be zero).

    Returns:
        Tensor: The value of Carlson's elliptic integral R_F.
    """
    return torchscience.ops.torchscience._carlson_elliptic_integral_r_e(x, y, z)
