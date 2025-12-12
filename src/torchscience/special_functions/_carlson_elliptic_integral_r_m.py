import torch
from torch import Tensor

import torchscience

__all__ = ["carlson_elliptic_integral_r_m"]


def carlson_elliptic_integral_r_m(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    r"""
    Carlson's elliptic integral combination :math:`R_M(x, y, z)`.

    .. math::
        R_M(x, y, z) = 2 R_G(x, y, z) - R_F(x, y, z)

    This is a useful combination of Carlson's symmetric forms R_G and R_F
    that appears in various elliptic integral identities. Named r_m as an
    alternative naming convention.

    Args:
        x (Tensor): First argument (must be non-negative).
        y (Tensor): Second argument (must be non-negative).
        z (Tensor): Third argument (must be non-negative, at most one can be zero).

    Returns:
        Tensor: The value of the elliptic integral combination R_M.
    """
    return torchscience.ops.torchscience._carlson_elliptic_integral_r_m(x, y, z)
