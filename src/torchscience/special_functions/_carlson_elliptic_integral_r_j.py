import torch
from torch import Tensor

import torchscience

__all__ = ["carlson_elliptic_integral_r_j"]


def carlson_elliptic_integral_r_j(x: Tensor, y: Tensor, z: Tensor, p: Tensor) -> Tensor:
    r"""
    Carlson's elliptic integral :math:`R_J(x, y, z, p)`.

    .. math::
        R_J(x, y, z, p) = \frac{3}{2} \int_0^{\infty} \frac{dt}{(t + p)\sqrt{(t + x)(t + y)(t + z)}}

    This is one of the Carlson symmetric forms of elliptic integrals.

    Args:
        x (Tensor): First argument (must be non-negative).
        y (Tensor): Second argument (must be non-negative).
        z (Tensor): Third argument (must be non-negative, at most one of x, y, z can be zero).
        p (Tensor): Fourth argument (must be positive).

    Returns:
        Tensor: The value of Carlson's elliptic integral R_J.
    """
    return torchscience.ops.torchscience._carlson_elliptic_integral_r_j(x, y, z, p)
