import torch
from torch import Tensor

import torchscience

__all__ = ["carlson_elliptic_integral_r_d"]


def carlson_elliptic_integral_r_d(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    r"""
    Carlson's elliptic integral :math:`R_D(x, y, z)`.

    .. math::
        R_D(x, y, z) = \frac{3}{2} \int_0^{\infty} \frac{dt}{(t + x)^{1/2}(t + y)^{1/2}(t + z)^{3/2}}

    This is related to other Carlson forms by:

    .. math::
        R_D(x, y, z) = R_J(x, y, z, z)

    Args:
        x (Tensor): First argument (must be non-negative).
        y (Tensor): Second argument (must be non-negative, at most one of x and y can be zero).
        z (Tensor): Third argument (must be positive).

    Returns:
        Tensor: The value of Carlson's elliptic integral R_D.
    """
    return torchscience.ops.torchscience._carlson_elliptic_integral_r_d(x, y, z)
