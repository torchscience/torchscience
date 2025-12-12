import torch
from torch import Tensor

import torchscience

__all__ = ["carlson_elliptic_integral_r_k"]


def carlson_elliptic_integral_r_k(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    r"""
    Carlson's elliptic integral :math:`R_G(x, y, z)`.

    .. math::
        R_G(x, y, z) = \frac{1}{4\pi} \int \sqrt{x l^2 + y m^2 + z n^2} \, d\Omega

    Or equivalently:

    .. math::
        R_G(x, y, z) = \frac{1}{4} \int_0^{\infty} \frac{1}{\sqrt{(t+x)(t+y)(t+z)}}
        \left(\frac{x}{t+x} + \frac{y}{t+y} + \frac{z}{t+z}\right) t \, dt

    This is Carlson's symmetric form R_G, named r_k as an alternative naming convention.

    Args:
        x (Tensor): First argument (must be non-negative).
        y (Tensor): Second argument (must be non-negative).
        z (Tensor): Third argument (must be non-negative, at most one can be zero).

    Returns:
        Tensor: The value of Carlson's elliptic integral R_G.
    """
    return torchscience.ops.torchscience._carlson_elliptic_integral_r_k(x, y, z)
