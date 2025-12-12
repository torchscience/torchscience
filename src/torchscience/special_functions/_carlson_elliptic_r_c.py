import torch
from torch import Tensor

import torchscience

__all__ = ["carlson_elliptic_r_c"]


def carlson_elliptic_r_c(x: Tensor, y: Tensor) -> Tensor:
    r"""
    Carlson's degenerate elliptic integral :math:`R_C(x, y)`.

    .. math::
        R_C(x, y) = \frac{1}{2} \int_0^{\infty} \frac{dt}{\sqrt{t + x}(t + y)}

    This is a special case of Carlson's elliptic integral :math:`R_F`:
    :math:`R_C(x, y) = R_F(x, y, y)`.

    Args:
        x (Tensor): Input tensor for x (must be non-negative).
        y (Tensor): Input tensor for y (must be non-zero).

    Returns:
        Tensor: The value of Carlson's degenerate elliptic integral R_C.
    """
    return torchscience.ops.torchscience._carlson_elliptic_r_c(x, y)
